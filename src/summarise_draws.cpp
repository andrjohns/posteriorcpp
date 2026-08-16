// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

static double quantile7(const std::vector<double>& sorted, double p) {
  const int n = sorted.size();
  const double h = (n - 1) * p;
  const int lo = (int) std::floor(h);
  const int hi = (int) std::ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

template <typename Derived>
static MatrixXd split_chains(const Eigen::MatrixBase<Derived>& x) {
  const int niter = x.rows();
  const int half = niter / 2;
  if (half == 0) return x;
  MatrixXd out(half, 2 * x.cols());
  out.leftCols(x.cols()) = x.topRows(half);
  out.rightCols(x.cols()) = x.bottomRows(half);
  return out;
}

static MatrixXd z_scale(const Ref<const MatrixXd>& x) {
  const int S = x.size();
  const double* const data = x.data();
  std::vector<int> idx(S);
  for (int i = 0; i < S; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return data[a] < data[b]; });
  std::vector<double> rank(S);
  int i = 0;
  while (i < S) {
    int j = i;
    while (j + 1 < S && data[idx[j + 1]] == data[idx[i]]) j++;
    const double avg_rank = (i + j) / 2.0 + 1.0;
    for (int k = i; k <= j; k++) rank[idx[k]] = avg_rank;
    i = j + 1;
  }
  constexpr double c = 3.0 / 8.0;
  const double denom = S - 2 * c + 1;
  MatrixXd z(x.rows(), x.cols());
  for (int k = 0; k < S; k++) z.data()[k] = R::qnorm((rank[k] - c) / denom, 0.0, 1.0, 1, 0);
  return z;
}

struct FFTWorkspace {
  Eigen::FFT<double> fft;
  std::vector<std::complex<double>> yc, freq, time;
};

static int nextn(int n) {
  for (int cand = n;; cand++) {
    int m = cand;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    if (m == 1) return cand;
  }
}

static void autocovariance(const Ref<const VectorXd>& x, FFTWorkspace& ws, Eigen::Ref<VectorXd> out) {
  const int N = x.size();
  const double xmean = x.mean();
  const double varx = (x.array() - xmean).square().sum() / (N - 1);
  if (varx == 0.0) {
    out.setZero();
    return;
  }
  const int M = nextn(N);
  const int Mt2 = 2 * M;
  ws.yc.assign(Mt2, std::complex<double>(0.0, 0.0));
  for (int i = 0; i < N; i++) ws.yc[i] = std::complex<double>(x[i] - xmean, 0.0);
  ws.fft.fwd(ws.freq, ws.yc);
  for (auto& z : ws.freq) z = std::complex<double>(std::norm(z), 0.0);
  ws.fft.inv(ws.time, ws.freq);
  for (int i = 0; i < N; i++) out[i] = ws.time[i].real();
  out *= varx * (N - 1) / N / out[0];
}

static double rhat_basic(const Ref<const MatrixXd>& x) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (!x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS) return NA_REAL;
  const VectorXd chain_mean = x.colwise().mean();
  double var_within = 0.0;
  for (int c = 0; c < nchains; c++)
    var_within += (x.col(c).array() - chain_mean[c]).square().sum() / (niter - 1);
  var_within /= nchains;
  const double mu = chain_mean.mean();
  const double var_between = niter * (chain_mean.array() - mu).square().sum() / (nchains - 1);
  return std::sqrt((var_between / var_within + niter - 1) / niter);
}

static double ess_basic(const Ref<const MatrixXd>& x, FFTWorkspace& ws) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (niter < 3 || !x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS) return NA_REAL;
  MatrixXd acov(niter, nchains);
  for (int c = 0; c < nchains; c++) autocovariance(x.col(c), ws, acov.col(c));
  const VectorXd acov_means = acov.rowwise().mean();
  const double mean_var = acov_means[0] * niter / (niter - 1.0);
  double var_plus = acov_means[0];
  if (nchains > 1) {
    const VectorXd chain_mean = x.colwise().mean();
    const double mu = chain_mean.mean();
    var_plus += (chain_mean.array() - mu).square().sum() / (nchains - 1);
  }
  std::vector<double> rho(niter, 0.0);
  int t = 0;
  double rho_even = 1.0;
  rho[0] = rho_even;
  double rho_odd = 1.0 - (mean_var - acov_means[1]) / var_plus;
  rho[1] = rho_odd;
  while (t < niter - 5 && !std::isnan(rho_even + rho_odd) && (rho_even + rho_odd > 0)) {
    t += 2;
    rho_even = 1.0 - (mean_var - acov_means[t]) / var_plus;
    rho_odd = 1.0 - (mean_var - acov_means[t + 1]) / var_plus;
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
  const double ess = (double) nchains * niter;
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

  tbb::parallel_for(tbb::blocked_range<int>(0, nvars), [&](const tbb::blocked_range<int>& range) {
    FFTWorkspace ws;
    std::vector<double> sorted(n), abs_dev(n);

    for (int v = range.begin(); v != range.end(); ++v) {
      const Eigen::Map<const MatrixXd> X(ptr + (std::size_t) v * n, niter, nchains);

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
      const double sd_val = std::sqrt((X.array() - mean_val).square().sum() / (n - 1));

      std::copy(X.data(), X.data() + n, sorted.begin());
      std::sort(sorted.begin(), sorted.end());
      const double median_val = quantile7(sorted, 0.5);
      const double q5_val = quantile7(sorted, 0.05);
      const double q95_val = quantile7(sorted, 0.95);

      for (int i = 0; i < n; i++) abs_dev[i] = std::fabs(sorted[i] - median_val);
      std::sort(abs_dev.begin(), abs_dev.end());
      const double mad_val = 1.4826 * quantile7(abs_dev, 0.5);

      const MatrixXd Xsz = z_scale(split_chains(X));
      const double rhat_bulk = rhat_basic(Xsz);
      const double ess_bulk_val = ess_basic(Xsz, ws);

      const MatrixXd Xfsz = z_scale(split_chains((X.array() - median_val).abs().matrix()));
      const double rhat_tail = rhat_basic(Xfsz);

      const bool x_bad = !X.allFinite() || (X.maxCoeff() - X.minCoeff()) < EPS;
      double ess_q5 = NA_REAL;
      double ess_q95 = NA_REAL;
      if (!x_bad) {
        ess_q5 = ess_basic(split_chains((X.array() <= q5_val).cast<double>().matrix()), ws);
        ess_q95 = ess_basic(split_chains((X.array() <= q95_val).cast<double>().matrix()), ws);
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
    Rcpp::Named("mean") = mean_out,
    Rcpp::Named("median") = median_out,
    Rcpp::Named("sd") = sd_out,
    Rcpp::Named("mad") = mad_out,
    Rcpp::Named("q5") = q5_out,
    Rcpp::Named("q95") = q95_out,
    Rcpp::Named("rhat") = rhat_out,
    Rcpp::Named("ess_bulk") = ess_bulk_out,
    Rcpp::Named("ess_tail") = ess_tail_out
  );
}
