// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppEigen.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <unsupported/Eigen/FFT>
#include <vector>

static constexpr double EPS = std::numeric_limits<double>::epsilon();

// std::min/std::max do not reliably propagate NaN (they return whichever
// argument compares "not less than" the other under `<`, and NaN
// comparisons are always false, so the result silently depends on argument
// order). R's min()/max() always return NA if either input is NA. These
// match R's semantics.
static double na_min(double a, double b) {
  return (std::isnan(a) || std::isnan(b)) ? NA_REAL : std::min(a, b);
}

static double na_max(double a, double b) {
  return (std::isnan(a) || std::isnan(b)) ? NA_REAL : std::max(a, b);
}

// A matrix is "degenerate" for rhat/ess purposes if it has non-finite
// entries or every entry is (numerically) identical.
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

static Eigen::MatrixXd z_scale(const Eigen::Ref<const Eigen::MatrixXd>& x) {
  const int S = x.size();
  const double* const data = x.data();
  // Sort contiguous (value, index) pairs rather than an index array that
  // dereferences `data` on every comparison — much better cache locality.
  std::vector<std::pair<double, int>> sorted(S);
  for (int i = 0; i < S; i++) {
    sorted[i] = {data[i], i};
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const std::pair<double, int>& a,
               const std::pair<double, int>& b) { return a.first < b.first; });
  std::vector<double> rank(S);
  int i = 0;
  while (i < S) {
    int j = i;
    while (j + 1 < S && sorted[j + 1].first == sorted[i].first) {
      j++;
    }
    const double avg_rank = (i + j) / 2.0 + 1.0;
    for (int k = i; k <= j; k++) {
      rank[sorted[k].second] = avg_rank;
    }
    i = j + 1;
  }
  constexpr double c = 3.0 / 8.0;
  const double denom = S - 2 * c + 1;
  Eigen::MatrixXd z(x.rows(), x.cols());
  for (int k = 0; k < S; k++) {
    z.data()[k] = R::qnorm((rank[k] - c) / denom, 0.0, 1.0, 1, 0);
  }
  return z;
}

// Next 5-smooth number >= n that is also a multiple of 4, so that kissfft's
// optimized real-FFT path (which requires nfft % 4 == 0) is always used.
static int nextn(int n) {
  for (int cand = n;; cand++) {
    if (cand % 2 != 0) {
      continue;
    }
    int m = cand;
    while (m % 2 == 0) {
      m /= 2;
    }
    while (m % 3 == 0) {
      m /= 3;
    }
    while (m % 5 == 0) {
      m /= 5;
    }
    if (m == 1) {
      return cand;
    }
  }
}

static void autocovariance(const Eigen::Ref<const Eigen::VectorXd>& x,
                           Eigen::FFT<double>& fft,
                           Eigen::Ref<Eigen::VectorXd> out) {
  const int N = x.size();
  const double xmean = x.mean();
  const double varx = (x.array() - xmean).square().sum() / (N - 1);
  if (varx == 0.0) {
    out.setZero();
    return;
  }
  const int M = nextn(N);
  const int Mt2 = 2 * M;
  // Real-input FFT: works on the Hermitian half-spectrum instead of a
  // zero-imaginary complex signal, roughly halving the FFT arithmetic.
  std::vector<double> re(Mt2, 0.0);
  std::vector<std::complex<double>> freq;
  for (int i = 0; i < N; i++) {
    re[i] = x[i] - xmean;
  }
  fft.fwd(freq, re);
  for (auto& z : freq) {
    z = std::complex<double>(std::norm(z), 0.0);
  }
  fft.inv(re, freq);
  for (int i = 0; i < N; i++) {
    out[i] = re[i];
  }
  out *= varx * (N - 1) / N / out[0];
}

static double rhat_basic(const Eigen::Ref<const Eigen::MatrixXd>& x) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (is_degenerate(x)) {
    return NA_REAL;
  }
  const Eigen::VectorXd chain_mean = x.colwise().mean();
  double var_within = 0.0;
  for (int c = 0; c < nchains; c++) {
    var_within +=
        (x.col(c).array() - chain_mean[c]).square().sum() / (niter - 1);
  }
  var_within /= nchains;
  const double mu = chain_mean.mean();
  const double var_between =
      niter * (chain_mean.array() - mu).square().sum() / (nchains - 1);
  return std::sqrt((var_between / var_within + niter - 1) / niter);
}

static double ess_basic(const Eigen::Ref<const Eigen::MatrixXd>& x,
                        Eigen::FFT<double>& fft) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (niter < 3 || is_degenerate(x)) {
    return NA_REAL;
  }
  Eigen::MatrixXd acov(niter, nchains);
  for (int c = 0; c < nchains; c++) {
    autocovariance(x.col(c), fft, acov.col(c));
  }
  const Eigen::VectorXd acov_means = acov.rowwise().mean();
  const double mean_var = acov_means[0] * niter / (niter - 1.0);
  double var_plus = acov_means[0];
  if (nchains > 1) {
    const Eigen::VectorXd chain_mean = x.colwise().mean();
    const double mu = chain_mean.mean();
    var_plus += (chain_mean.array() - mu).square().sum() / (nchains - 1);
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
  double sum_rho = 0.0;
  for (int k = 0; k < max_t; k++) {
    sum_rho += rho[k];
  }
  double tau_hat = -1.0 + 2.0 * sum_rho + rho[max_t];
  const double ess = static_cast<double>(nchains) * niter;
  const double tau_bound = 1.0 / std::log10(ess);
  if (tau_hat < tau_bound) {
    tau_hat = tau_bound;
  }
  return ess / tau_hat;
}

// `want` is a length-16 logical vector in the fixed canonical order: mean,
// median, sd, var, mad, q5, q95, rhat, rhat_basic, ess_bulk, ess_tail,
// ess_basic, ess_mean, ess_sd, mcse_mean, mcse_sd. Statistics that share
// underlying work (e.g. rhat and ess_bulk both need z_scale(split_chains(x));
// ess_mean is exactly ess_basic under another name; mcse_sd and ess_sd share
// the ESS of the squared centered draws) still only compute that shared work
// once; statistics that aren't requested skip their computation (and output
// column) entirely rather than being computed and discarded.
// [[Rcpp::export]]
Rcpp::List summarise_draws_cpp(Rcpp::NumericVector draws,
                               Rcpp::LogicalVector want) {
  const Rcpp::IntegerVector dim = draws.attr("dim");
  const int niter = dim[0];
  const int nchains = dim[1];
  const int nvars = dim[2];
  const double* const ptr = draws.begin();

  const bool want_mean = want[0];
  const bool want_median = want[1];
  const bool want_sd = want[2];
  const bool want_var = want[3];
  const bool want_mad = want[4];
  const bool want_q5 = want[5];
  const bool want_q95 = want[6];
  const bool want_rhat = want[7];
  const bool want_rhat_basic = want[8];
  const bool want_ess_bulk = want[9];
  const bool want_ess_tail = want[10];
  const bool want_ess_basic = want[11];
  const bool want_ess_mean = want[12];
  const bool want_ess_sd = want[13];
  const bool want_mcse_mean = want[14];
  const bool want_mcse_sd = want[15];

  // Sorting the flattened draws is only needed to derive median/q5/q95 (and
  // mad, and rhat's folded-tail split, and ess_tail's quantile indicators) —
  // if none of those are requested, skip the sort entirely.
  const bool need_sort = want_median || want_mad || want_q5 || want_q95 ||
                         want_rhat || want_ess_tail;
  // rhat_basic/ess_basic/ess_mean/mcse_mean all need split_chains(x)
  // unnormalised (ess_mean(x) is defined identically to ess_basic(x), and
  // mcse_mean needs that same ESS as its denominator); rhat/ess_bulk need it
  // rank-normalised (z_scale) on top. Both start from the same
  // split_chains(x), so that's only computed once for either.
  const bool need_raw_ess = want_ess_basic || want_ess_mean || want_mcse_mean;
  const bool need_raw_bulk_pass = want_rhat_basic || need_raw_ess;
  const bool need_norm_bulk_pass = want_rhat || want_ess_bulk;
  const bool need_sd_val = want_sd || want_var || want_mcse_mean;
  const bool need_ess_sd_val = want_ess_sd || want_mcse_sd;

  // Pre-filled with NA so a NaN-containing variable can just `continue`
  // without writing NA to every requested column individually below — every
  // other code path always overwrites its column explicitly.
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

  tbb::parallel_for(
      tbb::blocked_range<int>(0, nvars),
      [&](const tbb::blocked_range<int>& range) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        std::vector<double> sorted, abs_dev;
        if (need_sort) {
          sorted.resize(n);
        }
        if (want_mad) {
          abs_dev.resize(n);
        }

        for (int v = range.begin(); v != range.end(); ++v) {
          const Eigen::Map<const Eigen::MatrixXd> X(
              ptr + static_cast<std::size_t>(v) * n, niter, nchains);

          if (X.array().isNaN().any()) {
            continue;  // outputs already NA-filled
          }

          const double mean_val = X.mean();
          if (want_mean) {
            mean_out[v] = mean_val;
          }
          double sd_val = NA_REAL;
          if (need_sd_val) {
            const double var_val =
                (X.array() - mean_val).square().sum() / (n - 1);
            sd_val = std::sqrt(var_val);
            if (want_sd) {
              sd_out[v] = sd_val;
            }
            if (want_var) {
              var_out[v] = var_val;
            }
          }

          double median_val = NA_REAL, q5_val = NA_REAL, q95_val = NA_REAL;
          if (need_sort) {
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
            for (int i = 0; i < n; i++) {
              abs_dev[i] = std::fabs(sorted[i] - median_val);
            }
            std::sort(abs_dev.begin(), abs_dev.end());
            mad_out[v] = 1.4826 * quantile7(abs_dev, 0.5);
          }

          double rhat_bulk = NA_REAL;
          double ess_raw_val = NA_REAL;  // = ess_basic(x) = ess_mean(x)
          if (need_raw_bulk_pass || need_norm_bulk_pass) {
            const Eigen::MatrixXd Xs = split_chains(X);
            if (want_rhat_basic) {
              rhat_basic_out[v] = rhat_basic(Xs);
            }
            if (need_raw_ess) {
              ess_raw_val = ess_basic(Xs, fft);
              if (want_ess_basic) {
                ess_basic_out[v] = ess_raw_val;
              }
              if (want_ess_mean) {
                ess_mean_out[v] = ess_raw_val;
              }
            }
            if (need_norm_bulk_pass) {
              const Eigen::MatrixXd Xsz = z_scale(Xs);
              if (want_rhat) {
                rhat_bulk = rhat_basic(Xsz);
              }
              if (want_ess_bulk) {
                ess_bulk_out[v] = ess_basic(Xsz, fft);
              }
            }
          }
          if (want_mcse_mean) {
            mcse_mean_out[v] = sd_val / std::sqrt(ess_raw_val);
          }

          double rhat_tail = NA_REAL;
          if (want_rhat) {
            const Eigen::MatrixXd Xfsz =
                z_scale(split_chains((X.array() - median_val).abs().matrix()));
            rhat_tail = rhat_basic(Xfsz);
            rhat_out[v] = na_max(rhat_bulk, rhat_tail);
          }

          if (want_ess_tail) {
            double ess_q5 = NA_REAL;
            double ess_q95 = NA_REAL;
            if (!is_degenerate(X)) {
              ess_q5 = ess_basic(
                  split_chains((X.array() <= q5_val).cast<double>().matrix()),
                  fft);
              ess_q95 = ess_basic(
                  split_chains((X.array() <= q95_val).cast<double>().matrix()),
                  fft);
            }
            ess_tail_out[v] = na_min(ess_q5, ess_q95);
          }

          // ess_sd(x) = ess of the squared centered draws (unnormalised,
          // split); mcse_sd reuses that exact ESS as its own denominator, so
          // both are gated on the same underlying value.
          double ess_sd_val = NA_REAL;
          if (need_ess_sd_val) {
            const Eigen::MatrixXd centered_sq =
                (X.array() - mean_val).square().matrix();
            ess_sd_val = ess_basic(split_chains(centered_sq), fft);
            if (want_ess_sd) {
              ess_sd_out[v] = ess_sd_val;
            }
          }
          if (want_mcse_sd) {
            const auto centered = (X.array() - mean_val);
            const double e_var = centered.square().mean();
            const double e_var4 = centered.square().square().mean();
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
