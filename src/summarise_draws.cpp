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
#include <numeric>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

struct ThreadBuffers {
  Eigen::FFT<double> fft;
  std::vector<std::complex<double>> yc, freq, time;
  std::vector<int> idx;
  std::vector<int> split_idx;
  std::vector<double> sorted_dev;
  MatrixXd split_raw;
  MatrixXd split_dev;
  MatrixXd acov_mat;
  VectorXd acov_means;
  VectorXd chain_mean;
  VectorXd chain_var;
  VectorXd chain_sum_sq;
  std::vector<double> rho;
  MatrixXd z_buf;
  MatrixXd fsz_buf;
  MatrixXd i5_buf;
  MatrixXd i95_buf;
};

static inline int nextn(int n) {
  for (int cand = n;; cand++) {
    int m = cand;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    if (m == 1) return cand;
  }
}

// Compute autocovariance without allocating memory
static void autocovariance_fast(
    const Eigen::Ref<const VectorXd>& x,
    const int N,
    const int Mt2,
    ThreadBuffers& buf,
    Eigen::Ref<VectorXd> out) {
  const double xmean = x.mean();
  const double varx = (x.array() - xmean).square().sum() / (N - 1);
  if (varx == 0.0) {
    out.setZero();
    return;
  }
  buf.yc.assign(Mt2, std::complex<double>(0.0, 0.0));
  for (int i = 0; i < N; i++) {
    buf.yc[i] = std::complex<double>(x[i] - xmean, 0.0);
  }
  buf.fft.fwd(buf.freq, buf.yc);
  for (int i = 0; i < Mt2; i++) {
    buf.freq[i] = std::complex<double>(std::norm(buf.freq[i]), 0.0);
  }
  buf.fft.inv(buf.time, buf.freq);
  const double scale = (varx * (N - 1) / N) / buf.time[0].real();
  for (int i = 0; i < N; i++) {
    out[i] = buf.time[i].real() * scale;
  }
}

// Compute basic Rhat using vectorised operations and preallocated buffers
static double rhat_basic_fast(const Ref<const MatrixXd>& x, ThreadBuffers& buf) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (!x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS) return NA_REAL;

  buf.chain_mean.resize(nchains);
  buf.chain_var.resize(nchains);
  buf.chain_mean = x.colwise().mean();
  buf.chain_var = (x.rowwise() - buf.chain_mean.transpose()).colwise().squaredNorm() / (niter - 1);

  const double var_within = buf.chain_var.mean();
  const double mu = buf.chain_mean.mean();
  const double var_between = niter * (buf.chain_mean.array() - mu).square().sum() / (nchains - 1);
  return std::sqrt((var_between / var_within + niter - 1) / niter);
}

// Compute basic ESS using vectorised operations, zero allocations and precalculated FFT size
static double ess_basic_fast(
    const Ref<const MatrixXd>& x,
    const int Mt2,
    const double ess_val,
    const double tau_bound,
    ThreadBuffers& buf) {
  const int niter = x.rows();
  const int nchains = x.cols();
  if (niter < 3 || !x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS) return NA_REAL;

  buf.acov_mat.resize(niter, nchains);
  for (int c = 0; c < nchains; c++) {
    autocovariance_fast(x.col(c), niter, Mt2, buf, buf.acov_mat.col(c));
  }

  buf.acov_means.resize(niter);
  buf.acov_means = buf.acov_mat.rowwise().mean();
  const double mean_var = buf.acov_means[0] * niter / (niter - 1.0);
  double var_plus = buf.acov_means[0];
  if (nchains > 1) {
    buf.chain_mean.resize(nchains);
    buf.chain_mean = x.colwise().mean();
    const double mu = buf.chain_mean.mean();
    var_plus += (buf.chain_mean.array() - mu).square().sum() / (nchains - 1);
  }

  buf.rho.assign(niter, 0.0);
  int t = 0;
  double rho_even = 1.0;
  buf.rho[0] = rho_even;
  double rho_odd = 1.0 - (mean_var - buf.acov_means[1]) / var_plus;
  buf.rho[1] = rho_odd;
  while (t < niter - 5 && !std::isnan(rho_even + rho_odd) && (rho_even + rho_odd > 0)) {
    t += 2;
    rho_even = 1.0 - (mean_var - buf.acov_means[t]) / var_plus;
    rho_odd = 1.0 - (mean_var - buf.acov_means[t + 1]) / var_plus;
    if (rho_even + rho_odd >= 0) {
      buf.rho[t] = rho_even;
      buf.rho[t + 1] = rho_odd;
    }
  }
  const int max_t = t;
  if (rho_even > 0) buf.rho[max_t] = rho_even;
  t = 0;
  while (t <= max_t - 4) {
    t += 2;
    if (buf.rho[t] + buf.rho[t + 1] > buf.rho[t - 2] + buf.rho[t - 1]) {
      buf.rho[t] = (buf.rho[t - 2] + buf.rho[t - 1]) / 2.0;
      buf.rho[t + 1] = buf.rho[t];
    }
  }
  double sum_rho = 0.0;
  for (int k = 0; k < max_t; k++) sum_rho += buf.rho[k];
  double tau_hat = -1.0 + 2.0 * sum_rho + buf.rho[max_t];
  if (tau_hat < tau_bound) tau_hat = tau_bound;
  return ess_val / tau_hat;
}

// Compute rank normalization directly on input matrix x into z_dest
static void z_scale_matrix(
    const MatrixXd& x,
    const int S,
    const double c,
    const double denom,
    const std::vector<double>& qnorm_table,
    ThreadBuffers& buf,
    MatrixXd& z_dest) {
  const double* const data = x.data();
  std::iota(buf.split_idx.begin(), buf.split_idx.end(), 0);
  std::sort(buf.split_idx.begin(), buf.split_idx.end(), [&](int a, int b) {
    return data[a] < data[b];
  });

  double* const out_data = z_dest.data();
  int i = 0;
  while (i < S) {
    int j = i;
    while (j + 1 < S && data[buf.split_idx[j + 1]] == data[buf.split_idx[i]]) j++;
    if (i == j) {
      out_data[buf.split_idx[i]] = qnorm_table[i];
    } else {
      const double avg_rank = (i + j) / 2.0 + 1.0;
      const double val = R::qnorm((avg_rank - c) / denom, 0.0, 1.0, 1, 0);
      for (int k = i; k <= j; k++) {
        out_data[buf.split_idx[k]] = val;
      }
    }
    i = j + 1;
  }
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
  const int niter_split = half;
  const int nchains_split = 2 * nchains;
  const int offset2 = niter - half;

  // Precompute quantile7 parameters
  const double h_med = (n - 1) * 0.5;
  const int lo_med = (int) std::floor(h_med);
  const int hi_med = (int) std::ceil(h_med);
  const double w_med = h_med - lo_med;

  const double h_5 = (n - 1) * 0.05;
  const int lo_5 = (int) std::floor(h_5);
  const int hi_5 = (int) std::ceil(h_5);
  const double w_5 = h_5 - lo_5;

  const double h_95 = (n - 1) * 0.95;
  const int lo_95 = (int) std::floor(h_95);
  const int hi_95 = (int) std::ceil(h_95);
  const double w_95 = h_95 - lo_95;

  // Precompute constants for z_scale and ESS
  const int S_split = niter_split * nchains_split;
  constexpr double c_offset = 3.0 / 8.0;
  const double denom_offset = S_split - 2 * c_offset + 1.0;
  std::vector<double> qnorm_table(S_split);
  for (int i = 0; i < S_split; i++) {
    qnorm_table[i] = R::qnorm((i + 1 - c_offset) / denom_offset, 0.0, 1.0, 1, 0);
  }

  const int M_split = nextn(niter_split);
  const int Mt2_split = 2 * M_split;
  const double ess_total = (double) nchains_split * niter_split;
  const double tau_bound = 1.0 / std::log10(ess_total);

  tbb::parallel_for(tbb::blocked_range<int>(0, nvars), [&](const tbb::blocked_range<int>& range) {
    ThreadBuffers buf;
    buf.idx.resize(n);
    buf.split_idx.resize(S_split);
    buf.sorted_dev.resize(n);
    buf.split_raw.resize(niter_split, nchains_split);
    buf.split_dev.resize(niter_split, nchains_split);
    buf.z_buf.resize(niter_split, nchains_split);
    buf.fsz_buf.resize(niter_split, nchains_split);
    buf.i5_buf.resize(niter_split, nchains_split);
    buf.i95_buf.resize(niter_split, nchains_split);

    for (int v = range.begin(); v != range.end(); ++v) {
      const double* const x_ptr = ptr + (std::size_t) v * n;
      const Eigen::Map<const MatrixXd> X(x_ptr, niter, nchains);

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

      // Compute sorted indices once for quantile extraction
      std::iota(buf.idx.begin(), buf.idx.end(), 0);
      std::sort(buf.idx.begin(), buf.idx.end(), [&](int a, int b) {
        return x_ptr[a] < x_ptr[b];
      });

      const double q5_val = x_ptr[buf.idx[lo_5]] + w_5 * (x_ptr[buf.idx[hi_5]] - x_ptr[buf.idx[lo_5]]);
      const double median_val = x_ptr[buf.idx[lo_med]] + w_med * (x_ptr[buf.idx[hi_med]] - x_ptr[buf.idx[lo_med]]);
      const double q95_val = x_ptr[buf.idx[lo_95]] + w_95 * (x_ptr[buf.idx[hi_95]] - x_ptr[buf.idx[lo_95]]);

      // Save min and max for constant check before buf.idx is reused
      const double min_coeff = x_ptr[buf.idx[0]];
      const double max_coeff = x_ptr[buf.idx[n - 1]];

      // MAD calculation (use copy so folded draws array is not permuted)
      for (int i = 0; i < n; i++) {
        buf.sorted_dev[i] = std::fabs(x_ptr[i] - median_val);
      }
      std::nth_element(buf.sorted_dev.begin(), buf.sorted_dev.begin() + hi_med, buf.sorted_dev.end());
      const double val_hi = buf.sorted_dev[hi_med];
      const double val_lo = (lo_med == hi_med) ? val_hi : *std::max_element(buf.sorted_dev.begin(), buf.sorted_dev.begin() + hi_med);
      const double mad_val = 1.4826 * (val_lo + w_med * (val_hi - val_lo));

      // Populate split_raw and split_dev matrices in memory
      for (int ch = 0; ch < nchains; ch++) {
        const double* const col_src = x_ptr + ch * niter;
        double* const dest_raw_1 = buf.split_raw.data() + ch * niter_split;
        double* const dest_raw_2 = buf.split_raw.data() + (nchains + ch) * niter_split;
        double* const dest_dev_1 = buf.split_dev.data() + ch * niter_split;
        double* const dest_dev_2 = buf.split_dev.data() + (nchains + ch) * niter_split;

        for (int r = 0; r < niter_split; r++) {
          dest_raw_1[r] = col_src[r];
          dest_raw_2[r] = col_src[r + offset2];
          dest_dev_1[r] = std::fabs(col_src[r] - median_val);
          dest_dev_2[r] = std::fabs(col_src[r + offset2] - median_val);
        }
      }

      // Rank normalization for bulk
      z_scale_matrix(buf.split_raw, S_split, c_offset, denom_offset, qnorm_table, buf, buf.z_buf);
      const double rhat_bulk = rhat_basic_fast(buf.z_buf, buf);
      const double ess_bulk_val = ess_basic_fast(buf.z_buf, Mt2_split, ess_total, tau_bound, buf);

      // Rank normalization for tail (folded draws)
      z_scale_matrix(buf.split_dev, S_split, c_offset, denom_offset, qnorm_table, buf, buf.fsz_buf);
      const double rhat_tail = rhat_basic_fast(buf.fsz_buf, buf);

      // Split indicator matrices for q5 and q95
      const bool x_bad = !X.allFinite() || (max_coeff - min_coeff) < EPS;
      double ess_q5 = NA_REAL;
      double ess_q95 = NA_REAL;
      if (!x_bad) {
        for (int ch = 0; ch < nchains; ch++) {
          const double* const col_src = x_ptr + ch * niter;
          double* const dest5_1 = buf.i5_buf.data() + ch * niter_split;
          double* const dest5_2 = buf.i5_buf.data() + (nchains + ch) * niter_split;
          double* const dest95_1 = buf.i95_buf.data() + ch * niter_split;
          double* const dest95_2 = buf.i95_buf.data() + (nchains + ch) * niter_split;

          for (int r = 0; r < niter_split; r++) {
            dest5_1[r] = (col_src[r] <= q5_val) ? 1.0 : 0.0;
            dest95_1[r] = (col_src[r] <= q95_val) ? 1.0 : 0.0;
            dest5_2[r] = (col_src[r + offset2] <= q5_val) ? 1.0 : 0.0;
            dest95_2[r] = (col_src[r + offset2] <= q95_val) ? 1.0 : 0.0;
          }
        }
        ess_q5 = ess_basic_fast(buf.i5_buf, Mt2_split, ess_total, tau_bound, buf);
        ess_q95 = ess_basic_fast(buf.i95_buf, Mt2_split, ess_total, tau_bound, buf);
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
