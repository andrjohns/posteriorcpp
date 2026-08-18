#pragma once

#include <Rcpp.h>

// TBB parallelises over columns in each translation unit, so pocketfft's own
// pool stays off.
#define POCKETFFT_CACHE_SIZE 16
#define POCKETFFT_NO_MULTITHREADING
#include <pocketfft_hdronly.h>

// The vendored Eigen predates EIGEN_POCKETFFT_DEFAULT, so the backend header
// is added on top and selected as a template argument instead.
// clang-format off
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/src/FFT/pocketfft_impl.h>
// clang-format on

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

// Shared Eigen/TBB primitives for effective-sample-size estimation, used by
// both summarise_draws_cpp_() and loo_cpp_().
namespace posteriorcpp {

using PocketFFT = Eigen::FFT<double, Eigen::internal::pocketfft_impl<double>>;

inline constexpr double EPS = std::numeric_limits<double>::epsilon();

inline bool is_degenerate(const Eigen::Ref<const Eigen::MatrixXd>& x) {
  return !x.allFinite() || (x.maxCoeff() - x.minCoeff()) < EPS;
}

template <typename Derived>
inline Eigen::MatrixXd split_chains(const Eigen::MatrixBase<Derived>& x) {
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
inline int split_chains_size(int niter, int nchains) {
  return (niter / 2 == 0 ? niter : 2 * (niter / 2)) * nchains;
}

// Autocovariance of the centred series via the Wiener–Khinchin theorem:
// the inverse transform of the power spectrum |fwd(x)|^2.
inline void autocovariance(const Eigen::Ref<const Eigen::VectorXd>& x,
                           const double xmean,
                           Eigen::Ref<Eigen::VectorXd> out) {
  const std::size_t N = static_cast<std::size_t>(x.size());
  // Zero-pad to L = 2N so the circular FFT correlation becomes linear.
  const std::size_t L = pocketfft::detail::util::good_size_real(2 * N);
  // Aligned: a plain std::vector's start address varies call to call, which
  // flips squaredNorm()'s peel-loop length and its summation order.
  std::vector<double, Eigen::aligned_allocator<double>> real(L, 0.0);
  Eigen::Map<Eigen::VectorXd>(real.data(), N) = x.array() - xmean;
  const double ss =
      Eigen::Map<const Eigen::VectorXd>(real.data(), N).squaredNorm();
  if (ss == 0.0) {
    out.setZero();
    return;
  }
  std::vector<std::complex<double>,
              Eigen::aligned_allocator<std::complex<double>>>
      spec(L / 2 + 1);
  // Unscaled drops a 1/L pass the normalisation below makes redundant.
  PocketFFT fft;
  fft.SetFlag(PocketFFT::HalfSpectrum);
  fft.SetFlag(PocketFFT::Unscaled);
  fft.fwd(spec.data(), real.data(), static_cast<int>(L));
  Eigen::Map<Eigen::ArrayXcd> S(spec.data(), static_cast<int>(spec.size()));
  S = S.abs2();
  fft.inv(real.data(), spec.data(), static_cast<int>(L));
  out.head(N) = Eigen::Map<const Eigen::VectorXd>(real.data(), N);
  // Normalise: varx * (N - 1) == ss, so the (N - 1) factors cancel exactly.
  out *= (ss / static_cast<double>(N)) / out[0];
}

// Effective sample size (Geyer's initial monotone sequence estimator) for a
// matrix whose columns are (possibly split/transformed) chains.
inline double ess_basic(const Eigen::Ref<const Eigen::MatrixXd>& x,
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
  // Aligned for the same reason as autocovariance()'s scratch buffers above.
  std::vector<double, Eigen::aligned_allocator<double>> rho(niter, 0.0);
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
  // In the reference implementation, rho_hat_t[1:0] selects the first
  // element (R ignores the zero index). Preserve that boundary behaviour for
  // split chains of length 3--5, where max_t remains zero.
  const int sum_count = std::max(max_t, 1);
  double sum_rho =
      Eigen::Map<const Eigen::VectorXd>(rho.data(), sum_count).sum();
  double tau_hat = -1.0 + 2.0 * sum_rho + rho[max_t];
  const double ess = static_cast<double>(nchains) * niter;
  const double tau_bound = 1.0 / std::log10(ess);
  if (tau_hat < tau_bound) {
    tau_hat = tau_bound;
  }
  return ess / tau_hat;
}

}  // namespace posteriorcpp
