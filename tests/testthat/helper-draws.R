make_draws <- function(niter, nchains, nvars, seed = NULL, fun = stats::rnorm) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  posterior::as_draws_array(
    array(fun(niter * nchains * nvars), dim = c(niter, nchains, nvars))
  )
}

all_stats <- c(
  "mean",
  "median",
  "sd",
  "mad",
  "q5",
  "q95",
  "rhat",
  "ess_bulk",
  "ess_tail"
)

all_supported_stats <- c(
  "mean",
  "median",
  "sd",
  "var",
  "mad",
  "q5",
  "q95",
  "rhat",
  "rhat_basic",
  "ess_bulk",
  "ess_tail",
  "ess_basic",
  "ess_mean",
  "ess_sd",
  "mcse_mean",
  "mcse_sd"
)

reference_stat <- function(stat, x) {
  switch(
    stat,
    mean = mean(as.vector(x)),
    median = median(as.vector(x)),
    sd = sd(as.vector(x)),
    var = var(as.vector(x)),
    mad = mad(as.vector(x)),
    q5 = unname(quantile(as.vector(x), 0.05, type = 7)),
    q95 = unname(quantile(as.vector(x), 0.95, type = 7)),
    rhat = posterior::rhat(x),
    rhat_basic = posterior::rhat_basic(x),
    ess_bulk = posterior::ess_bulk(x),
    ess_tail = posterior::ess_tail(x),
    ess_basic = posterior::ess_basic(x),
    ess_mean = posterior::ess_mean(x),
    ess_sd = posterior::ess_sd(x),
    mcse_mean = posterior::mcse_mean(x),
    mcse_sd = posterior::mcse_sd(x)
  )
}

stat_tolerance <- function(stat) {
  if (startsWith(stat, "ess_") || startsWith(stat, "mcse_")) {
    # Different FFT backends can perturb autocovariances enough to change a
    # discrete positive-sequence truncation point on rare inputs.
    1e-4
  } else {
    1e-13
  }
}

expect_matches_posterior <- function(x, stats = NULL, tolerance = NULL) {
  a <- if (is.null(stats)) {
    posteriorcpp::summarise_draws_cpp(x)
  } else {
    posteriorcpp::summarise_draws_cpp(x, stats = stats)
  }
  b <- posterior::summarise_draws(x)

  a <- a[order(a$variable), ]
  b <- b[order(b$variable), ]

  expect_identical(a$variable, b$variable)

  requested <- if (is.null(stats)) {
    all_stats
  } else {
    all_stats[all_stats %in% stats]
  }
  expect_identical(names(a), c("variable", requested))

  for (cn in requested) {
    current_tolerance <- if (is.null(tolerance)) {
      stat_tolerance(cn)
    } else {
      tolerance
    }
    expect_equal(
      a[[cn]],
      b[[cn]],
      tolerance = current_tolerance,
      info = paste("column:", cn)
    )
  }
  invisible(a)
}

expect_matches_direct_references <- function(x, stats = all_supported_stats) {
  out <- posteriorcpp::summarise_draws_cpp(x, stats = stats)
  vars <- posterior::variables(x)

  for (stat in stats) {
    expected <- vapply(vars, function(v) {
      reference_stat(stat, posterior::extract_variable_matrix(x, v))
    }, numeric(1))
    expect_equal(
      out[[stat]],
      unname(expected),
      tolerance = stat_tolerance(stat),
      info = paste("direct reference column:", stat)
    )
  }
  invisible(out)
}
