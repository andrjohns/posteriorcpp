# Shared helpers for comparing bayescpp::summarise_draws() against
# posterior::summarise_draws() across many draws shapes and stat subsets.

make_draws <- function(niter, nchains, nvars, seed = NULL, fun = stats::rnorm) {
  if (!is.null(seed)) set.seed(seed)
  posterior::as_draws_array(
    array(fun(niter * nchains * nvars), dim = c(niter, nchains, nvars))
  )
}

all_stats <- c("mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail")

# Compares bayescpp::summarise_draws(x, stats) against posterior's default
# (single-core) summary, column by column. Tolerance is 1e-4 rather than
# something tighter because bayescpp's real-input FFT and posterior's
# complex-to-complex FFT accumulate floating point error differently; for
# almost all variables this differs at the 1e-10-1e-12 level, but the
# rhat/ess rho-truncation recursion (Geyer's initial monotone sequence) has
# discrete branch conditions, and on rare variables sitting right at a
# branch boundary that FFT-level noise can flip a truncation decision,
# producing a "different but still numerically valid" result on the order of
# 1e-5. 1e-4 comfortably absorbs that while remaining far tighter than any
# genuine correctness bug would produce.
expect_matches_posterior <- function(x, stats = NULL, tolerance = 1e-4) {
  a <- if (is.null(stats)) {
    bayescpp::summarise_draws(x)
  } else {
    bayescpp::summarise_draws(x, stats = stats)
  }
  b <- posterior::summarise_draws(x)

  a <- a[order(a$variable), ]
  b <- b[order(b$variable), ]

  expect_identical(a$variable, b$variable)

  requested <- if (is.null(stats)) all_stats else all_stats[all_stats %in% stats]
  expect_identical(names(a), c("variable", requested))

  for (cn in requested) {
    expect_equal(a[[cn]], b[[cn]], tolerance = tolerance, info = paste("column:", cn))
  }
  invisible(a)
}
