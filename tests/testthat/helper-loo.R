make_log_lik <- function(niter, nchains, nobs, seed = NULL, sd = 1.5) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  array(rnorm(niter * nchains * nobs, sd = sd), dim = c(niter, nchains, nobs))
}

# loo::relative_eff(exp(x)) can underflow-degenerate for very-negative log-lik
# columns; shifting first (as loo_cpp() does internally) avoids that and is
# what should be compared against for an exact match.
shifted_relative_eff <- function(x) {
  loo::relative_eff(exp(sweep(x, 3, apply(x, 3, max), FUN = "-")))
}

expect_matches_loo <- function(
  x,
  r_eff = TRUE,
  tolerance = 1e-8,
  check_log_weights = TRUE
) {
  ref_r_eff <- if (isTRUE(r_eff)) {
    shifted_relative_eff(x)
  } else if (isFALSE(r_eff) || is.null(r_eff)) {
    1
  } else {
    r_eff
  }
  ref <- suppressWarnings(loo::loo(x, r_eff = ref_r_eff, save_psis = TRUE))
  mine <- suppressWarnings(loo_cpp(x, r_eff = r_eff, save_psis = TRUE))

  expect_equal(
    unclass(mine$estimates),
    unclass(ref$estimates),
    tolerance = tolerance
  )
  expect_equal(
    mine$pointwise[, "elpd_loo"],
    ref$pointwise[, "elpd_loo"],
    tolerance = tolerance
  )
  expect_equal(
    mine$pointwise[, "p_loo"],
    ref$pointwise[, "p_loo"],
    tolerance = tolerance
  )
  expect_equal(
    mine$pointwise[, "looic"],
    ref$pointwise[, "looic"],
    tolerance = tolerance
  )
  expect_equal(
    mine$pointwise[, "mcse_elpd_loo"],
    ref$pointwise[, "mcse_elpd_loo"],
    tolerance = tolerance
  )
  expect_equal(
    unname(mine$pointwise[, "influence_pareto_k"]),
    unname(ref$diagnostics$pareto_k),
    tolerance = tolerance
  )
  expect_equal(
    mine$diagnostics$pareto_k,
    ref$diagnostics$pareto_k,
    tolerance = tolerance
  )
  expect_equal(
    mine$diagnostics$n_eff,
    ref$diagnostics$n_eff,
    tolerance = tolerance
  )
  expect_equal(
    mine$diagnostics$r_eff,
    ref$diagnostics$r_eff,
    tolerance = tolerance
  )
  if (check_log_weights) {
    expect_equal(
      mine$psis_object$log_weights,
      ref$psis_object$log_weights,
      tolerance = tolerance
    )
  }
  invisible(mine)
}
