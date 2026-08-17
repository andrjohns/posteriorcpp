make_draws <- function(niter, nchains, nvars, seed = NULL, fun = stats::rnorm) {
  if (!is.null(seed)) set.seed(seed)
  posterior::as_draws_array(
    array(fun(niter * nchains * nvars), dim = c(niter, nchains, nvars))
  )
}

all_stats <- c("mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail")

# Tolerance 1e-4 absorbs FFT noise that can flip a discrete rho-truncation
# branch on rare variables, giving a different but valid result around 1e-5.
expect_matches_posterior <- function(x, stats = NULL, tolerance = 1e-4) {
  a <- if (is.null(stats)) {
    posteriorcpp::summarise_draws_cpp(x)
  } else {
    posteriorcpp::summarise_draws_cpp(x, stats = stats)
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
