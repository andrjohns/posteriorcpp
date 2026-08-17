// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppEigen.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// autocovariance() uses pocketfft's r2c()/c2r(), whose plans are cached
// internally (POCKETFFT_CACHE_SIZE entries). TBB parallelises over variables
// here, so pocketfft's own worker pool is disabled (nthreads=1 below).
#define POCKETFFT_CACHE_SIZE 16
#define POCKETFFT_NO_MULTITHREADING
#include "include/pocketfft_hdronly.h"

static constexpr double EPS = std::numeric_limits<double>::epsilon();

// std::min/max don't propagate NaN, R's min/max do.
static double na_min(double a, double b) {
  return (std::isnan(a) || std::isnan(b)) ? NA_REAL : std::min(a, b);
}

static double na_max(double a, double b) {
  return (std::isnan(a) || std::isnan(b)) ? NA_REAL : std::max(a, b);
}

static bool is_degenerate(const Eigen::Ref<const Eigen::MatrixXd>& x) {
  return !x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS;
}

static double quantile7(const std::vector<double>& sorted, double p) {
  const int n = sorted.size();
  const double h = (n - 1) * p;
  const int lo = static_cast<int>(std::floor(h));
  const int hi = static_cast<int>(std::ceil(h));
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

template <typename Derived>
static Eigen::MatrixXd split_chains(const Eigen::MatrixBase<Derived>& x) {
  const int niter = x.rows();
  const int half = niter / 2;
  if (half == 0) {
    return x;
  }
  Eigen::MatrixXd out(half, 2 * x.cols());
  out.leftCols(x.cols()) = x.topRows(half);
  out.rightCols(x.cols()) = x.bottomRows(half);
  return out;
}

// split_chains(X)'s element count -- odd niter drops one row, so this is
// niter*nchains only when niter is even.
static inline int split_chains_size(int niter, int nchains) {
  return (niter / 2 == 0 ? niter : 2 * (niter / 2)) * nchains;
}

struct ZScratch {
  std::vector<double> rank;
  std::vector<std::pair<double, int>> pbuf;  // sort scratch
};

// 1-based ranks with ties averaged, written to sc.rank in input order.
static void average_ranks(const double* data, int S, ZScratch& sc) {
  sc.pbuf.resize(S);
  for (int i = 0; i < S; i++) {
    sc.pbuf[i] = {data[i], i};
  }
  std::sort(
      sc.pbuf.begin(), sc.pbuf.end(),
      [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
      });
  sc.rank.resize(S);
  int i = 0;
  while (i < S) {
    int j = i;
    while (j + 1 < S && sc.pbuf[j + 1].first == sc.pbuf[i].first) {
      j++;
    }
    const double avg_rank = (i + j) / 2.0 + 1.0;
    for (int k = i; k <= j; k++) {
      sc.rank[sc.pbuf[k].second] = avg_rank;
    }
    i = j + 1;
  }
}

// S depends only on (niter, nchains), so it's identical across every
// variable, thread, and repeat call with that shape -- cache it globally.
static std::mutex g_qnorm_lut_mutex;
static std::unordered_map<int, std::shared_ptr<const std::vector<double>>>
    g_qnorm_lut_cache;

static std::shared_ptr<const std::vector<double>> qnorm_lut(int S) {
  {
    std::lock_guard<std::mutex> lock(g_qnorm_lut_mutex);
    const auto it = g_qnorm_lut_cache.find(S);
    if (it != g_qnorm_lut_cache.end()) {
      return it->second;
    }
  }
  constexpr double c = 3.0 / 8.0;
  const double denom = S - 2 * c + 1;
  auto lut = std::make_shared<std::vector<double>>(S);
  for (int k = 0; k < S; k++) {
    (*lut)[k] = R::qnorm((k + 1 - c) / denom, 0.0, 1.0, 1, 0);
  }
  std::lock_guard<std::mutex> lock(g_qnorm_lut_mutex);
  return g_qnorm_lut_cache.emplace(S, std::move(lut)).first->second;
}

static Eigen::MatrixXd z_scale(const Eigen::Ref<const Eigen::MatrixXd>& x,
                               ZScratch& sc, const std::vector<double>& lut) {
  const int S = x.size();
  average_ranks(x.data(), S, sc);
  constexpr double c = 3.0 / 8.0;
  const double denom = S - 2 * c + 1;
  // Untied ranks are exactly 1..S, served by the shared lut. Tied ranks
  // average to half-integers and fall through to the direct call.
  Eigen::MatrixXd z(x.rows(), x.cols());
  for (int k = 0; k < S; k++) {
    const double r = sc.rank[k];
    const int ir = static_cast<int>(r);
    z.data()[k] =
        (r == ir) ? lut[ir - 1] : R::qnorm((r - c) / denom, 0.0, 1.0, 1, 0);
  }
  return z;
}

// Autocovariance of the centred series via the Wiener–Khinchin theorem:
// the inverse transform of the power spectrum |r2c(x)|^2.
static void autocovariance(const Eigen::Ref<const Eigen::VectorXd>& x,
                           const double xmean, Eigen::Ref<Eigen::VectorXd> out) {
  const std::size_t N = static_cast<std::size_t>(x.size());
  // Zero-pad to L = 2N so the circular FFT correlation becomes linear.
  const std::size_t L =
      pocketfft::detail::util::good_size_real(2 * N);
  std::vector<double> real(L, 0.0);
  Eigen::Map<Eigen::VectorXd>(real.data(), N) = x.array() - xmean;
  const double ss =
      Eigen::Map<const Eigen::VectorXd>(real.data(), N).squaredNorm();
  if (ss == 0.0) {
    out.setZero();
    return;
  }
  std::vector<std::complex<double>> spec(L / 2 + 1);
  const pocketfft::shape_t shape{L};
  const pocketfft::stride_t stride{sizeof(double)};
  const pocketfft::stride_t cstride{sizeof(std::complex<double>)};
  pocketfft::r2c<double>(shape, stride, cstride, 0, true, real.data(),
                         spec.data(), 1.0);
  for (auto& c : spec) {
    c = c * std::conj(c);
  }
  pocketfft::c2r<double>(shape, cstride, stride, 0, false, spec.data(),
                         real.data(), 1.0);
  out.head(N) = Eigen::Map<const Eigen::VectorXd>(real.data(), N);
  // Normalise: varx * (N - 1) == ss, so the (N - 1) factors cancel exactly.
  out *= (ss / static_cast<double>(N)) / out[0];
}

static double rhat_basic(const Eigen::Ref<const Eigen::MatrixXd>& x,
                         const Eigen::VectorXd* chain_mean = nullptr) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (chain_mean == nullptr) {
    // Callers that supply chain_mean have already vetted this matrix.
    if (is_degenerate(x)) {
      return NA_REAL;
    }
  }
  const Eigen::VectorXd cm = chain_mean ? *chain_mean : x.colwise().mean();
  // cm must be transposed to a row vector for the rowwise broadcast.
  const double var_within =
      (x.array().rowwise() - cm.array().transpose()).matrix().squaredNorm() /
      (niter - 1) / nchains;
  const double mu = cm.mean();
  // Unscaled by niter: B = niter * chain_var, and that niter cancels the
  // trailing division in (B/W + niter - 1)/niter.
  const double chain_var =
      (cm.array() - mu).matrix().squaredNorm() / (nchains - 1);
  return std::sqrt(chain_var / var_within + (niter - 1.0) / niter);
}

static double ess_basic(const Eigen::Ref<const Eigen::MatrixXd>& x,
                        const Eigen::VectorXd* chain_mean) {
  const int niter = x.rows();
  const int nchains = x.cols();
  // A non-null chain_mean means the caller already vetted this matrix.
  if (niter < 3 || (chain_mean == nullptr && is_degenerate(x))) {
    return NA_REAL;
  }
  const Eigen::VectorXd cm =
      chain_mean ? Eigen::VectorXd(*chain_mean) : x.colwise().mean();
  Eigen::MatrixXd acov(niter, nchains);
  for (int c = 0; c < nchains; c++) {
    autocovariance(x.col(c), cm[c], acov.col(c));
  }
  const Eigen::VectorXd acov_means = acov.rowwise().mean();
  const double mean_var = acov_means[0] * niter / (niter - 1.0);
  double var_plus = acov_means[0];
  if (nchains > 1) {
    const double mu = cm.mean();
    var_plus += (cm.array() - mu).matrix().squaredNorm() / (nchains - 1);
  }
  std::vector<double> rho(niter, 0.0);
  int t = 0;
  double rho_even = 1.0;
  rho[0] = rho_even;
  double rho_odd = 1.0 - (mean_var - acov_means[1]) / var_plus;
  rho[1] = rho_odd;
  while (t < niter - 5 && !std::isnan(rho_even + rho_odd) &&
         (rho_even + rho_odd > 0)) {
    t += 2;
    rho_even = 1.0 - (mean_var - acov_means[t]) / var_plus;
    rho_odd = 1.0 - (mean_var - acov_means[t + 1]) / var_plus;
    if (rho_even + rho_odd >= 0) {
      rho[t] = rho_even;
      rho[t + 1] = rho_odd;
    }
  }
  const int max_t = t;
  if (rho_even > 0) {
    rho[max_t] = rho_even;
  }
  t = 0;
  while (t <= max_t - 4) {
    t += 2;
    if (rho[t] + rho[t + 1] > rho[t - 2] + rho[t - 1]) {
      rho[t] = (rho[t - 2] + rho[t - 1]) / 2.0;
      rho[t + 1] = rho[t];
    }
  }
  double sum_rho = Eigen::Map<const Eigen::VectorXd>(rho.data(), max_t).sum();
  double tau_hat = -1.0 + 2.0 * sum_rho + rho[max_t];
  const double ess = static_cast<double>(nchains) * niter;
  const double tau_bound = 1.0 / std::log10(ess);
  if (tau_hat < tau_bound) {
    tau_hat = tau_bound;
  }
  return ess / tau_hat;
}

static bool has_stat(const Rcpp::CharacterVector& stats, const char* name) {
  for (R_xlen_t i = 0; i < stats.size(); i++) {
    if (stats[i] == name) {
      return true;
    }
  }
  return false;
}

// Shared work is computed once; unrequested stats skip computation entirely.
// [[Rcpp::export]]
Rcpp::List summarise_draws_cpp_(Rcpp::NumericVector draws,
                                Rcpp::CharacterVector stats) {
  const Rcpp::IntegerVector dim = draws.attr("dim");
  const int niter = dim[0];
  const int nchains = dim[1];
  const int nvars = dim[2];
  const double* const ptr = draws.begin();

  const bool want_mean = has_stat(stats, "mean");
  const bool want_median = has_stat(stats, "median");
  const bool want_sd = has_stat(stats, "sd");
  const bool want_var = has_stat(stats, "var");
  const bool want_mad = has_stat(stats, "mad");
  const bool want_q5 = has_stat(stats, "q5");
  const bool want_q95 = has_stat(stats, "q95");
  const bool want_rhat = has_stat(stats, "rhat");
  const bool want_rhat_basic = has_stat(stats, "rhat_basic");
  const bool want_ess_bulk = has_stat(stats, "ess_bulk");
  const bool want_ess_tail = has_stat(stats, "ess_tail");
  const bool want_ess_basic = has_stat(stats, "ess_basic");
  const bool want_ess_mean = has_stat(stats, "ess_mean");
  const bool want_ess_sd = has_stat(stats, "ess_sd");
  const bool want_mcse_mean = has_stat(stats, "mcse_mean");
  const bool want_mcse_sd = has_stat(stats, "mcse_sd");

  // Only mad needs a fully sorted copy (its in-place merge needs total order);
  // the scalar quantiles of larger inputs are served by selection.
  const bool need_sorted = want_median || want_mad || want_q5 || want_q95 ||
                           want_rhat || want_ess_tail;
  // All bulk stats share split_chains(x); rhat/ess_bulk add z_scale on top.
  const bool need_raw_ess = want_ess_basic || want_ess_mean || want_mcse_mean;
  const bool need_raw_bulk_pass = want_rhat_basic || need_raw_ess;
  const bool need_norm_bulk_pass = want_rhat || want_ess_bulk;
  const bool need_var_val =
      want_sd || want_var || want_mcse_mean || want_mcse_sd;
  const bool need_ess_sd_val = want_ess_sd || want_mcse_sd;
  // Degeneracy survives split/z_scale/abs/indicator, so one range check on X
  // serves every rhat/ess path.
  const bool need_x_degenerate = want_ess_tail || want_rhat_basic ||
                                 want_rhat || want_ess_bulk || need_raw_ess ||
                                 need_ess_sd_val;

  // Pre-filled with NA so NaN variables can `continue` without per-column
  // writes.
  Rcpp::NumericVector mean_out, median_out, sd_out, var_out, mad_out, q5_out,
      q95_out, rhat_out, rhat_basic_out, ess_bulk_out, ess_tail_out,
      ess_basic_out, ess_mean_out, ess_sd_out, mcse_mean_out, mcse_sd_out;
  if (want_mean) {
    mean_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_median) {
    median_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_sd) {
    sd_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_var) {
    var_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_mad) {
    mad_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_q5) {
    q5_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_q95) {
    q95_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_rhat) {
    rhat_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_rhat_basic) {
    rhat_basic_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_ess_bulk) {
    ess_bulk_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_ess_tail) {
    ess_tail_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_ess_basic) {
    ess_basic_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_ess_mean) {
    ess_mean_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_ess_sd) {
    ess_sd_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_mcse_mean) {
    mcse_mean_out = Rcpp::NumericVector(nvars, NA_REAL);
  }
  if (want_mcse_sd) {
    mcse_sd_out = Rcpp::NumericVector(nvars, NA_REAL);
  }

  const int n = niter * nchains;

  // z_scale() is the only qnorm_lut() consumer; skip building it otherwise.
  const bool need_z_scale = want_rhat || want_ess_bulk;
  const std::shared_ptr<const std::vector<double>> shared_qnorm_lut =
      need_z_scale ? qnorm_lut(split_chains_size(niter, nchains)) : nullptr;

  // auto_partitioner splits to one variable per chunk, so the body below runs
  // per variable, not per thread; scratch must live out here to be reused.
  struct Scratch {
    std::vector<double> sorted, abs_dev;
    ZScratch zsc;
    bool init = false;
  };
  tbb::enumerable_thread_specific<Scratch> scratch;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, nvars),
      [&](const tbb::blocked_range<int>& range) {
        Scratch& sc = scratch.local();
        if (!sc.init) {
          if (need_sorted) {
            sc.sorted.resize(n);
          }
          if (want_mad) {
            sc.abs_dev.resize(n);
          }
          sc.init = true;
        }
        std::vector<double>& sorted = sc.sorted;
        std::vector<double>& abs_dev = sc.abs_dev;
        ZScratch& zsc = sc.zsc;

        for (int v = range.begin(); v != range.end(); ++v) {
          const Eigen::Map<const Eigen::MatrixXd> X(
              ptr + static_cast<std::size_t>(v) * n, niter, nchains);

          if (X.array().isNaN().any()) {
            continue;
          }

          // Inf is left to the allFinite checks inside rhat_basic/ess_basic:
          // an infinite range reads as non-degenerate here.
          bool x_degenerate = false;
          if (need_x_degenerate) {
            x_degenerate = (X.maxCoeff() - X.minCoeff()) < EPS;
          }

          const double mean_val = X.mean();
          if (want_mean) {
            mean_out[v] = mean_val;
          }
          double sd_val = NA_REAL;
          double var_val = NA_REAL;
          double ss = NA_REAL;  // sum of squared centred draws
          if (need_var_val) {
            ss = (X.array() - mean_val).matrix().squaredNorm();
            var_val = ss / (n - 1);
            if (want_sd || want_var || want_mcse_mean) {
              sd_val = std::sqrt(var_val);
            }
            if (want_sd) {
              sd_out[v] = sd_val;
            }
            if (want_var) {
              var_out[v] = var_val;
            }
          }

          double median_val = NA_REAL, q5_val = NA_REAL, q95_val = NA_REAL;
          if (need_sorted) {
            std::copy(X.data(), X.data() + n, sorted.begin());
            std::sort(sorted.begin(), sorted.end());
            median_val = quantile7(sorted, 0.5);
            q5_val = quantile7(sorted, 0.05);
            q95_val = quantile7(sorted, 0.95);
            if (want_median) {
              median_out[v] = median_val;
            }
            if (want_q5) {
              q5_out[v] = q5_val;
            }
            if (want_q95) {
              q95_out[v] = q95_val;
            }
          }

          if (want_mad) {
            // Read outwards from the median both deviation halves already
            // ascend, so one linear merge replaces a full re-sort.
            const auto middle =
                std::lower_bound(sorted.begin(), sorted.end(), median_val);
            const int n_left = static_cast<int>(middle - sorted.begin());
            Eigen::Map<Eigen::ArrayXd> dev(abs_dev.data(), n);
            dev.head(n_left) = median_val - Eigen::Map<const Eigen::ArrayXd>(
                                                sorted.data(), n_left)
                                                .reverse();
            dev.tail(n - n_left) = Eigen::Map<const Eigen::ArrayXd>(
                                       sorted.data() + n_left, n - n_left) -
                                   median_val;
            std::inplace_merge(dev.begin(), dev.begin() + n_left, dev.end());
            mad_out[v] = 1.4826 * quantile7(abs_dev, 0.5);
          }

          double rhat_bulk = NA_REAL;
          double ess_raw_val = NA_REAL;
          // Degenerate variables are NA for every rhat/ess stat, and the
          // outputs are pre-filled with NA.
          if (!x_degenerate && (need_raw_bulk_pass || need_norm_bulk_pass)) {
            const Eigen::MatrixXd Xs = split_chains(X);
            if (need_raw_bulk_pass) {
              double rhat_basic_val = NA_REAL;
              if (want_rhat_basic && need_raw_ess) {
                // Paired: one degeneracy check and one chain-means pass
                // serve both stats.
                if (!is_degenerate(Xs)) {
                  const Eigen::VectorXd cm = Xs.colwise().mean();
                  rhat_basic_val = rhat_basic(Xs, &cm);
                  ess_raw_val = ess_basic(Xs, &cm);
                }
              } else {
                if (want_rhat_basic) {
                  rhat_basic_val = rhat_basic(Xs);
                }
                if (need_raw_ess) {
                  ess_raw_val = ess_basic(Xs, nullptr);
                }
              }
              if (want_rhat_basic) {
                rhat_basic_out[v] = rhat_basic_val;
              }
              if (need_raw_ess) {
                if (want_ess_basic) {
                  ess_basic_out[v] = ess_raw_val;
                }
                if (want_ess_mean) {
                  ess_mean_out[v] = ess_raw_val;
                }
              }
            }
            if (need_norm_bulk_pass) {
              const Eigen::MatrixXd Xsz = z_scale(Xs, zsc, *shared_qnorm_lut);
              if (want_rhat && want_ess_bulk) {
                // Paired: one degeneracy check and one chain-means pass
                // serve both stats.
                if (!is_degenerate(Xsz)) {
                  const Eigen::VectorXd cm = Xsz.colwise().mean();
                  rhat_bulk = rhat_basic(Xsz, &cm);
                  ess_bulk_out[v] = ess_basic(Xsz, &cm);
                }
              } else {
                if (want_rhat) {
                  rhat_bulk = rhat_basic(Xsz);
                }
                if (want_ess_bulk) {
                  ess_bulk_out[v] = ess_basic(Xsz, nullptr);
                }
              }
            }
          }
          if (want_mcse_mean) {
            mcse_mean_out[v] = sd_val / std::sqrt(ess_raw_val);
          }

          double rhat_tail = NA_REAL;
          if (want_rhat && !x_degenerate) {
            const Eigen::MatrixXd Xfsz =
                z_scale(split_chains((X.array() - median_val).abs().matrix()),
                        zsc, *shared_qnorm_lut);
            rhat_tail = rhat_basic(Xfsz);
            rhat_out[v] = na_max(rhat_bulk, rhat_tail);
          }

          if (want_ess_tail) {
            double ess_q5 = NA_REAL;
            double ess_q95 = NA_REAL;
            if (!x_degenerate) {
              ess_q5 = ess_basic(
                  split_chains((X.array() <= q5_val).cast<double>().matrix()),
                  nullptr);
              ess_q95 = ess_basic(
                  split_chains((X.array() <= q95_val).cast<double>().matrix()),
                  nullptr);
            }
            ess_tail_out[v] = na_min(ess_q5, ess_q95);
          }

          // ess_sd = ESS of squared centered draws; mcse_sd reuses it.
          double ess_sd_val = NA_REAL;
          if (need_ess_sd_val && !x_degenerate) {
            ess_sd_val = ess_basic(
                split_chains((X.array() - mean_val).square().matrix()), nullptr);
            if (want_ess_sd) {
              ess_sd_out[v] = ess_sd_val;
            }
          }
          if (want_mcse_sd) {
            // ss was already computed in the sd/var block above.
            const double e_var = ss / n;
            const double e_var4 =
                (X.array() - mean_val).square().square().mean();
            const double var_of_var = (e_var4 - e_var * e_var) / ess_sd_val;
            const double var_of_sd = var_of_var / e_var / 4.0;
            mcse_sd_out[v] = std::sqrt(var_of_sd);
          }
        }
      });

  Rcpp::List out;
  if (want_mean) {
    out["mean"] = mean_out;
  }
  if (want_median) {
    out["median"] = median_out;
  }
  if (want_sd) {
    out["sd"] = sd_out;
  }
  if (want_var) {
    out["var"] = var_out;
  }
  if (want_mad) {
    out["mad"] = mad_out;
  }
  if (want_q5) {
    out["q5"] = q5_out;
  }
  if (want_q95) {
    out["q95"] = q95_out;
  }
  if (want_rhat) {
    out["rhat"] = rhat_out;
  }
  if (want_rhat_basic) {
    out["rhat_basic"] = rhat_basic_out;
  }
  if (want_ess_bulk) {
    out["ess_bulk"] = ess_bulk_out;
  }
  if (want_ess_tail) {
    out["ess_tail"] = ess_tail_out;
  }
  if (want_ess_basic) {
    out["ess_basic"] = ess_basic_out;
  }
  if (want_ess_mean) {
    out["ess_mean"] = ess_mean_out;
  }
  if (want_ess_sd) {
    out["ess_sd"] = ess_sd_out;
  }
  if (want_mcse_mean) {
    out["mcse_mean"] = mcse_mean_out;
  }
  if (want_mcse_sd) {
    out["mcse_sd"] = mcse_sd_out;
  }
  return out;
}
