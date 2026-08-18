#' Fast PSIS-LOO cross-validation
#'
#' A C++/Eigen implementation of `loo::loo.array()`: approximate
#' leave-one-out cross-validation via Pareto-smoothed importance sampling
#' (PSIS).
#'
#' @param x A log-likelihood array with dimensions iterations x chains x
#'   observations, as returned by e.g.
#'   `posterior::extract_variable_matrix()`-style `draws_array` objects.
#' @param r_eff Relative MCMC effective sample sizes of `exp(log_lik)` for
#'   each observation, used to adjust the PSIS tail-length and diagnostics.
#'   One of:
#'   * `TRUE` (the default): estimate `r_eff` from `x` inside the same pass
#'     that computes PSIS-LOO, reusing the FFT-based ESS machinery instead of
#'     the separate [loo::relative_eff()] traversal of the array that
#'     `loo::loo()` callers otherwise need.
#'   * `FALSE` or `NULL`: skip the estimate and use `r_eff = 1` for every
#'     observation, matching `loo::loo()`'s own default.
#'   * A single number or a numeric vector with one value per observation,
#'     used as-is.
#' @param save_psis Should the smoothed, unnormalized PSIS log weights be
#'   kept in the returned object as `psis_object`? Matches the `save_psis`
#'   argument of [loo::loo()].
#' @return An object with the same structure as [loo::loo()]'s return value:
#'   class `c("psis_loo", "importance_sampling_loo", "loo")`, with
#'   `estimates`, `pointwise`, `diagnostics`, and (if `save_psis = TRUE`)
#'   `psis_object` components.
#' @export
loo_cpp <- function(x, r_eff = TRUE, save_psis = FALSE) {
  dim_x <- dim(x)
  if (length(dim_x) != 3L) {
    stop(
      "`x` must be a 3-D log-likelihood array (iterations x chains x observations)."
    )
  }
  n_obs <- dim_x[3]
  s <- dim_x[1] * dim_x[2]

  if (anyNA(x)) {
    stop("NAs not allowed in `x`.")
  }
  if (any(x == -Inf)) {
    # -x (the PSIS log ratio) would be +Inf, breaking the tail sort below.
    stop("`x` must not contain -Inf (a zero-likelihood draw breaks PSIS-LOO).")
  }

  auto_r_eff <- isTRUE(r_eff)
  if (auto_r_eff || isFALSE(r_eff) || is.null(r_eff)) {
    r_eff_vec <- rep(1, n_obs)
  } else if (length(r_eff) == 1L) {
    r_eff_vec <- rep(as.double(r_eff), n_obs)
  } else if (length(r_eff) == n_obs) {
    r_eff_vec <- as.double(r_eff)
  } else {
    stop(
      "`r_eff` must be TRUE/FALSE, a scalar, or a vector with one value per observation."
    )
  }
  if (!auto_r_eff && anyNA(r_eff_vec)) {
    message("Replacing NAs in `r_eff` with 1s")
    r_eff_vec[is.na(r_eff_vec)] <- 1
  }

  save_psis <- isTRUE(save_psis)
  raw <- loo_cpp_(unclass(x), r_eff_vec, auto_r_eff, save_psis)

  k_threshold <- min(1 - 1 / log10(s), 0.7)
  if (isTRUE(any(raw$pareto_k > k_threshold))) {
    warning(
      "Some Pareto k diagnostic values are too high. ",
      "See help('pareto-k-diagnostic') for details.\n",
      call. = FALSE
    )
  }

  pointwise <- cbind(
    elpd_loo = raw$elpd_loo,
    mcse_elpd_loo = raw$mcse_elpd_loo,
    p_loo = raw$p_loo,
    looic = raw$looic,
    influence_pareto_k = raw$pareto_k
  )

  summary_cols <- c("elpd_loo", "p_loo", "looic")
  summary_mat <- pointwise[, summary_cols, drop = FALSE]
  estimates <- cbind(
    Estimate = colSums(summary_mat),
    SE = sqrt(n_obs * apply(summary_mat, 2, stats::var))
  )
  rownames(estimates) <- summary_cols

  diagnostics <- list(
    pareto_k = raw$pareto_k,
    n_eff = raw$n_eff,
    r_eff = raw$r_eff
  )

  psis_object <- NULL
  if (save_psis) {
    psis_object <- structure(
      list(log_weights = raw$log_weights, diagnostics = diagnostics),
      norm_const_log = raw$norm_const_log,
      tail_len = raw$tail_len,
      r_eff = raw$r_eff,
      dims = c(s, n_obs),
      method = "psis",
      class = c("psis", "importance_sampling", "list")
    )
  }

  out <- list(
    estimates = estimates,
    pointwise = pointwise,
    diagnostics = diagnostics,
    psis_object = psis_object,

    # deprecated but kept for compatibility with the `loo` package API
    elpd_loo = estimates["elpd_loo", "Estimate"],
    p_loo = estimates["p_loo", "Estimate"],
    looic = estimates["looic", "Estimate"],
    se_elpd_loo = estimates["elpd_loo", "SE"],
    se_p_loo = estimates["p_loo", "SE"],
    se_looic = estimates["looic", "SE"]
  )

  structure(
    out,
    dims = c(s, n_obs),
    class = c("psis_loo", "importance_sampling_loo", "loo")
  )
}
