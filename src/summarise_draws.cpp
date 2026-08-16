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

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

static double quantile7(const std::vector<double>& sorted, double p) {
  const int n = sorted.size();
  const double h = (n - 1) * p;
  const int lo = (int)std::floor(h);
  const int hi = (int)std::ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

// Per-thread scratch space. All buffers below are sized once per thread
// (before the per-variable loop starts, since niter/nchains are constant
// across variables) and then overwritten in place on every variable, so a
// single summarise_draws_cpp() call does a handful of allocations per thread
// instead of dozens of heap allocations per variable.
struct Workspace {
  Eigen::FFT<double> fft;
  std::vector<double> fft_re;
  std::vector<std::complex<double>> fft_freq;

  std::vector<std::pair<double, int>> rank_pairs;
  std::vector<double> rank_vec;

  MatrixXd split_buf;
  MatrixXd z_buf;
  MatrixXd acov_buf;
  std::vector<double> rho_buf;
  VectorXd acov_means_buf;
  VectorXd chain_mean_buf;

  Workspace() { fft.SetFlag(Eigen::FFT<double>::HalfSpectrum); }
};

template <typename Derived>
static void split_chains_into(const Eigen::MatrixBase<Derived>& x,
                              Ref<MatrixXd> out) {
  const int niter = x.rows();
  const int half = niter / 2;
  if (half == 0) {
    out = x;
    return;
  }
  out.leftCols(x.cols()) = x.topRows(half);
  out.rightCols(x.cols()) = x.bottomRows(half);
}

static void z_scale_into(const Ref<const MatrixXd>& x, Workspace& ws,
                         Ref<MatrixXd> out) {
  const int S = x.size();
  const double* const data = x.data();
  // Sort contiguous (value, index) pairs rather than an index array that
  // dereferences `data` on every comparison — much better cache locality.
  ws.rank_pairs.resize(S);
  for (int i = 0; i < S; i++) ws.rank_pairs[i] = {data[i], i};
  std::sort(ws.rank_pairs.begin(), ws.rank_pairs.end(),
            [](const std::pair<double, int>& a,
               const std::pair<double, int>& b) { return a.first < b.first; });
  ws.rank_vec.resize(S);
  int i = 0;
  while (i < S) {
    int j = i;
    while (j + 1 < S && ws.rank_pairs[j + 1].first == ws.rank_pairs[i].first)
      j++;
    const double avg_rank = (i + j) / 2.0 + 1.0;
    for (int k = i; k <= j; k++)
      ws.rank_vec[ws.rank_pairs[k].second] = avg_rank;
    i = j + 1;
  }
  constexpr double c = 3.0 / 8.0;
  const double denom = S - 2 * c + 1;
  for (int k = 0; k < S; k++)
    out.data()[k] = R::qnorm((ws.rank_vec[k] - c) / denom, 0.0, 1.0, 1, 0);
}

// Next 5-smooth number >= n that is also a multiple of 4, so that kissfft's
// optimized real-FFT path (which requires nfft % 4 == 0) is always used.
static int nextn(int n) {
  for (int cand = n;; cand++) {
    if (cand % 2 != 0) continue;
    int m = cand;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    if (m == 1) return cand;
  }
}

static void autocovariance(const Ref<const VectorXd>& x, Workspace& ws,
                           Eigen::Ref<VectorXd> out) {
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
  ws.fft_re.assign(Mt2, 0.0);
  for (int i = 0; i < N; i++) ws.fft_re[i] = x[i] - xmean;
  ws.fft.fwd(ws.fft_freq, ws.fft_re);
  for (auto& z : ws.fft_freq) z = std::complex<double>(std::norm(z), 0.0);
  ws.fft.inv(ws.fft_re, ws.fft_freq);
  for (int i = 0; i < N; i++) out[i] = ws.fft_re[i];
  out *= varx * (N - 1) / N / out[0];
}

static double rhat_basic(const Ref<const MatrixXd>& x, Workspace& ws) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (!x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS) return NA_REAL;
  ws.chain_mean_buf = x.colwise().mean();
  double var_within = 0.0;
  for (int c = 0; c < nchains; c++)
    var_within +=
        (x.col(c).array() - ws.chain_mean_buf[c]).square().sum() / (niter - 1);
  var_within /= nchains;
  const double mu = ws.chain_mean_buf.mean();
  const double var_between =
      niter * (ws.chain_mean_buf.array() - mu).square().sum() / (nchains - 1);
  return std::sqrt((var_between / var_within + niter - 1) / niter);
}

static double ess_basic(const Ref<const MatrixXd>& x, Workspace& ws) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (niter < 3 || !x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS)
    return NA_REAL;
  for (int c = 0; c < nchains; c++)
    autocovariance(x.col(c), ws, ws.acov_buf.col(c));
  ws.acov_means_buf = ws.acov_buf.rowwise().mean();
  const double mean_var = ws.acov_means_buf[0] * niter / (niter - 1.0);
  double var_plus = ws.acov_means_buf[0];
  if (nchains > 1) {
    ws.chain_mean_buf = x.colwise().mean();
    const double mu = ws.chain_mean_buf.mean();
    var_plus += (ws.chain_mean_buf.array() - mu).square().sum() / (nchains - 1);
  }
  std::fill(ws.rho_buf.begin(), ws.rho_buf.end(), 0.0);
  std::vector<double>& rho = ws.rho_buf;
  int t = 0;
  double rho_even = 1.0;
  rho[0] = rho_even;
  double rho_odd = 1.0 - (mean_var - ws.acov_means_buf[1]) / var_plus;
  rho[1] = rho_odd;
  while (t < niter - 5 && !std::isnan(rho_even + rho_odd) &&
         (rho_even + rho_odd > 0)) {
    t += 2;
    rho_even = 1.0 - (mean_var - ws.acov_means_buf[t]) / var_plus;
    rho_odd = 1.0 - (mean_var - ws.acov_means_buf[t + 1]) / var_plus;
    if (rho_even + rho_odd >= 0) {
      rho[t] = rho_even;
      rho[t + 1] = rho_odd;
    }
  }
  const int max_t = t;
  if (rho_even > 0) rho[max_t] = rho_even;
  t = 0;
  while (t <= max_t - 4) {
    t += 2;
    if (rho[t] + rho[t + 1] > rho[t - 2] + rho[t - 1]) {
      rho[t] = (rho[t - 2] + rho[t - 1]) / 2.0;
      rho[t + 1] = rho[t];
    }
  }
  double sum_rho = 0.0;
  for (int k = 0; k < max_t; k++) sum_rho += rho[k];
  double tau_hat = -1.0 + 2.0 * sum_rho + rho[max_t];
  const double ess = (double)nchains * niter;
  const double tau_bound = 1.0 / std::log10(ess);
  if (tau_hat < tau_bound) tau_hat = tau_bound;
  return ess / tau_hat;
}

// [[Rcpp::export]]
Rcpp::List summarise_draws_cpp(Rcpp::NumericVector draws) {
  const Rcpp::IntegerVector dim = draws.attr("dim");
  const int niter = dim[0];
  const int nchains = dim[1];
  const int nvars = dim[2];
  const double* const ptr = draws.begin();

  Rcpp::NumericVector mean_out(nvars);
  Rcpp::NumericVector median_out(nvars);
  Rcpp::NumericVector sd_out(nvars);
  Rcpp::NumericVector mad_out(nvars);
  Rcpp::NumericVector q5_out(nvars);
  Rcpp::NumericVector q95_out(nvars);
  Rcpp::NumericVector rhat_out(nvars);
  Rcpp::NumericVector ess_bulk_out(nvars);
  Rcpp::NumericVector ess_tail_out(nvars);

  const int n = niter * nchains;
  const int half = niter / 2;
  const int srows = (half > 0) ? half : niter;
  const int scols = (half > 0) ? 2 * nchains : nchains;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, nvars),
      [&](const tbb::blocked_range<int>& range) {
        Workspace ws;
        ws.split_buf.resize(srows, scols);
        ws.z_buf.resize(srows, scols);
        ws.acov_buf.resize(srows, scols);
        ws.rho_buf.assign(srows, 0.0);
        ws.acov_means_buf.resize(srows);
        ws.chain_mean_buf.resize(scols);
        ws.rank_pairs.resize((std::size_t)srows * scols);
        ws.rank_vec.resize((std::size_t)srows * scols);
        std::vector<double> sorted(n), abs_dev(n);

        for (int v = range.begin(); v != range.end(); ++v) {
          const Eigen::Map<const MatrixXd> X(ptr + (std::size_t)v * n, niter,
                                             nchains);

          if (X.array().isNaN().any()) {
            mean_out[v] = NA_REAL;
            median_out[v] = NA_REAL;
            sd_out[v] = NA_REAL;
            mad_out[v] = NA_REAL;
            q5_out[v] = NA_REAL;
            q95_out[v] = NA_REAL;
            rhat_out[v] = NA_REAL;
            ess_bulk_out[v] = NA_REAL;
            ess_tail_out[v] = NA_REAL;
            continue;
          }

          const double mean_val = X.mean();
          const double sd_val =
              std::sqrt((X.array() - mean_val).square().sum() / (n - 1));

          std::copy(X.data(), X.data() + n, sorted.begin());
          std::sort(sorted.begin(), sorted.end());
          const double median_val = quantile7(sorted, 0.5);
          const double q5_val = quantile7(sorted, 0.05);
          const double q95_val = quantile7(sorted, 0.95);

          for (int i = 0; i < n; i++)
            abs_dev[i] = std::fabs(sorted[i] - median_val);
          std::sort(abs_dev.begin(), abs_dev.end());
          const double mad_val = 1.4826 * quantile7(abs_dev, 0.5);

          split_chains_into(X, ws.split_buf);
          z_scale_into(ws.split_buf, ws, ws.z_buf);
          const double rhat_bulk = rhat_basic(ws.z_buf, ws);
          const double ess_bulk_val = ess_basic(ws.z_buf, ws);

          split_chains_into((X.array() - median_val).abs().matrix(),
                            ws.split_buf);
          z_scale_into(ws.split_buf, ws, ws.z_buf);
          const double rhat_tail = rhat_basic(ws.z_buf, ws);

          const bool x_bad =
              !X.allFinite() || (X.maxCoeff() - X.minCoeff()) < EPS;
          double ess_q5 = NA_REAL;
          double ess_q95 = NA_REAL;
          if (!x_bad) {
            split_chains_into((X.array() <= q5_val).cast<double>().matrix(),
                              ws.split_buf);
            ess_q5 = ess_basic(ws.split_buf, ws);

            split_chains_into((X.array() <= q95_val).cast<double>().matrix(),
                              ws.split_buf);
            ess_q95 = ess_basic(ws.split_buf, ws);
          }

          mean_out[v] = mean_val;
          median_out[v] = median_val;
          sd_out[v] = sd_val;
          mad_out[v] = mad_val;
          q5_out[v] = q5_val;
          q95_out[v] = q95_val;
          rhat_out[v] = std::max(rhat_bulk, rhat_tail);
          ess_bulk_out[v] = ess_bulk_val;
          ess_tail_out[v] = std::min(ess_q5, ess_q95);
        }
      });

  return Rcpp::List::create(
      Rcpp::Named("mean") = mean_out, Rcpp::Named("median") = median_out,
      Rcpp::Named("sd") = sd_out, Rcpp::Named("mad") = mad_out,
      Rcpp::Named("q5") = q5_out, Rcpp::Named("q95") = q95_out,
      Rcpp::Named("rhat") = rhat_out, Rcpp::Named("ess_bulk") = ess_bulk_out,
      Rcpp::Named("ess_tail") = ess_tail_out);
}
