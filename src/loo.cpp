// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "internal.h"

using namespace posteriorcpp;

namespace {

// Generalized Pareto quantile function, location fixed at 0 (loo always
// fits/queries the shifted exceedances, never the raw tail).
double qgeneralized_pareto(double p, double sigma, double k) {
  if (k == 0.0) {
    return -sigma * std::log1p(-p);
  }
  return sigma * std::expm1(-k * std::log1p(-p)) / k;
}

double log_sum_exp(const Eigen::Ref<const Eigen::VectorXd>& x) {
  const double m = x.maxCoeff();
  if (m == -std::numeric_limits<double>::infinity()) {
    return m;
  }
  return m + std::log((x.array() - m).exp().sum());
}

struct GpdFit {
  double k;
  double sigma;
};

// Zhang & Stephens (2009) grid MLE, unweighted; `x` is pre-sorted ascending
// exceedances over the threshold.
GpdFit gpdfit(const Eigen::Ref<const Eigen::VectorXd>& x) {
  const int N = static_cast<int>(x.size());
  constexpr int kMinGridPts = 30;
  constexpr double kPrior = 3.0;
  const int M = kMinGridPts +
                static_cast<int>(std::floor(std::sqrt(static_cast<double>(N))));
  const double xstar = x[static_cast<int>(std::floor(N / 4.0 + 0.5)) - 1];
  if (!(xstar > x[0])) {
    // Sample too far from a generalized Pareto shape to fit reliably.
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  const double xn = x[N - 1];
  const auto xa = x.array();  // view, not a copy -- x already owns its data
  const double m_d = static_cast<double>(M);
  const Eigen::ArrayXd theta =
      1.0 / xn +
      (1.0 - (m_d * (Eigen::ArrayXd::LinSpaced(M, 1.0, m_d) - 0.5).inverse())
                 .sqrt()) /
          kPrior / xstar;
  const Eigen::ArrayXd k =
      theta.unaryExpr([&](double th) { return (-th * xa).log1p().mean(); });
  const Eigen::ArrayXd l_theta =
      static_cast<double>(N) * ((-theta / k).log() - k - 1.0);
  const double lse = log_sum_exp(l_theta.matrix());
  const double theta_hat = (theta * (l_theta - lse).exp()).sum();
  double k_hat = (-theta_hat * xa).log1p().mean();
  double sigma_hat = -k_hat / theta_hat;
  // Weakly-informative prior centred on k=0.5, stabilising the estimate at
  // small tail sample sizes (Vehtari et al. 2024).
  k_hat = (k_hat * N + 5.0) / (N + 10.0);
  if (std::isnan(k_hat)) {
    k_hat = std::numeric_limits<double>::infinity();
    sigma_hat = std::numeric_limits<double>::quiet_NaN();
  }
  return {k_hat, sigma_hat};
}

// Tail length used to fit the GPD: 3*sqrt(S)/r_eff, capped at 20% of draws.
int n_pareto(double r_eff, int S) {
  const double s_d = static_cast<double>(S);
  return static_cast<int>(
      std::ceil(std::min(0.2 * s_d, 3.0 * std::sqrt(s_d / r_eff))));
}

struct PsisResult {
  Eigen::VectorXd log_weights;  // smoothed, unnormalized, length S
  double pareto_k;
};

// Pareto-smoothed importance sampling for one observation's log ratios
// (-log_lik). Mirrors loo:::do_psis_i().
template <typename Derived>
PsisResult do_psis_i(const Eigen::MatrixBase<Derived>& log_ratios,
                     int tail_len) {
  const int S = static_cast<int>(log_ratios.size());
  const double max_lr = log_ratios.maxCoeff();
  Eigen::VectorXd lw = log_ratios.array() - max_lr;
  double khat = std::numeric_limits<double>::infinity();

  if (tail_len >= 5) {
    std::vector<std::pair<double, int>> ord(S);
    for (int i = 0; i < S; i++) {
      ord[i] = {lw[i], i};
    }
    std::sort(
        ord.begin(), ord.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          return a.first < b.first;
        });
    const int tail_start = S - tail_len;
    const double tail_min = ord[tail_start].first;
    const double tail_max = ord[S - 1].first;
    // Ties across the whole tail leave nothing to fit -- keep it unsmoothed.
    if (tail_max - tail_min >= EPS / 100.0) {
      const double cutoff = ord[tail_start - 1].first;
      const double exp_cutoff = std::exp(cutoff);
      const Eigen::VectorXd exceedances =
          Eigen::VectorXd::NullaryExpr(tail_len, [&](Eigen::Index j) {
            return std::exp(ord[tail_start + j].first) - exp_cutoff;
          });
      const GpdFit fit = gpdfit(exceedances);
      khat =
          std::isnan(fit.k) ? std::numeric_limits<double>::infinity() : fit.k;
      if (std::isfinite(fit.k)) {
        const Eigen::VectorXd smoothed =
            Eigen::VectorXd::NullaryExpr(tail_len, [&](Eigen::Index j) {
              const double p = (j + 0.5) / tail_len;
              return std::log(qgeneralized_pareto(p, fit.sigma, fit.k) +
                              exp_cutoff);
            });
        for (int j = 0; j < tail_len; j++) {
          lw[ord[tail_start + j].second] = smoothed[j];
        }
      }
    }
  }

  // Truncate at the max of the raw ratios (0, post-shift), then unshift.
  lw = (lw.array() > 0.0).select(max_lr, lw.array() + max_lr).matrix();
  return {std::move(lw), khat};
}

}  // namespace

// Mirrors loo::loo.array() (method = "psis"). auto_r_eff reuses the FFT ESS
// pass instead of a separate loo::relative_eff() traversal.
// [[Rcpp::export]]
Rcpp::List loo_cpp_(Rcpp::NumericVector log_lik, Rcpp::NumericVector r_eff,
                    bool auto_r_eff, bool save_psis) {
  const Rcpp::IntegerVector dim = log_lik.attr("dim");
  const int niter = dim[0];
  const int nchains = dim[1];
  const int nobs = dim[2];
  const double* const ptr = log_lik.begin();
  const int S = niter * nchains;

  Rcpp::NumericVector elpd_loo(nobs), mcse_elpd_loo(nobs), p_loo(nobs),
      looic(nobs), pareto_k(nobs), n_eff(nobs), r_eff_out(nobs),
      norm_const_log_out(nobs);
  Rcpp::IntegerVector tail_len_out(nobs);
  Rcpp::NumericMatrix log_weights_out;
  double* lw_ptr = nullptr;
  if (save_psis) {
    log_weights_out = Rcpp::NumericMatrix(S, nobs);
    lw_ptr = log_weights_out.begin();
  }

  tbb::parallel_for(
      tbb::blocked_range<int>(0, nobs),
      [&](const tbb::blocked_range<int>& range) {
        for (int v = range.begin(); v != range.end(); ++v) {
          const Eigen::Map<const Eigen::MatrixXd> X(
              ptr + static_cast<std::size_t>(v) * S, niter, nchains);
          const Eigen::Map<const Eigen::VectorXd> ll(X.data(), S);

          double r_eff_v = r_eff[v];
          if (auto_r_eff) {
            const double shift = X.maxCoeff();
            const double ess = ess_basic(
                split_chains((X.array() - shift).exp().matrix()), nullptr);
            r_eff_v = std::isfinite(ess) ? ess / S : 1.0;
          }
          r_eff_out[v] = r_eff_v;

          const int tail_len = n_pareto(r_eff_v, S);
          tail_len_out[v] = tail_len;
          PsisResult psis = do_psis_i(-ll, tail_len);
          pareto_k[v] = psis.pareto_k;

          const double norm_const_log = log_sum_exp(psis.log_weights);
          norm_const_log_out[v] = norm_const_log;
          const Eigen::VectorXd nlw = psis.log_weights.array() - norm_const_log;

          const double elpd_loo_v = log_sum_exp(ll + nlw);
          const double lpd_v =
              log_sum_exp(ll) - std::log(static_cast<double>(S));
          elpd_loo[v] = elpd_loo_v;
          p_loo[v] = lpd_v - elpd_loo_v;
          looic[v] = -2.0 * elpd_loo_v;

          const Eigen::ArrayXd w2 = nlw.array().exp().square();
          const double e_epd = std::exp(elpd_loo_v);
          const double var_epd =
              (w2 * (ll.array().exp() - e_epd).square()).sum() / r_eff_v;
          mcse_elpd_loo[v] = std::sqrt(std::log1p(var_epd / (e_epd * e_epd)));

          n_eff[v] = r_eff_v / w2.sum();

          if (save_psis) {
            std::copy(psis.log_weights.data(), psis.log_weights.data() + S,
                      lw_ptr + static_cast<std::size_t>(v) * S);
          }
        }
      });

  Rcpp::List out;
  out["elpd_loo"] = elpd_loo;
  out["mcse_elpd_loo"] = mcse_elpd_loo;
  out["p_loo"] = p_loo;
  out["looic"] = looic;
  out["pareto_k"] = pareto_k;
  out["n_eff"] = n_eff;
  out["r_eff"] = r_eff_out;
  out["norm_const_log"] = norm_const_log_out;
  out["tail_len"] = tail_len_out;
  if (save_psis) {
    out["log_weights"] = log_weights_out;
  }
  return out;
}
