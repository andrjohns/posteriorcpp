test_that("folded R-hat uses R-compatible even-sample median rounding", {
  set.seed(471)
  values <- rnorm(50 * 4) + sample(c(-4, 4), 50 * 4, replace = TRUE)
  x <- posterior::as_draws_array(array(values, dim = c(50, 1, 4)))

  out <- posteriorcpp::summarise_draws_cpp(x, stats = c("median", "rhat"))
  expected_median <- vapply(seq_len(4), function(v) {
    median(values[seq.int(1 + (v - 1) * 50, v * 50)])
  }, numeric(1))
  expected_rhat <- vapply(posterior::variables(x), function(v) {
    posterior::rhat(posterior::extract_variable_matrix(x, v))
  }, numeric(1))

  expect_identical(out$median, expected_median)
  expect_equal(out$rhat, unname(expected_rhat), tolerance = 1e-14)
  expect_equal(out$rhat[[1]], 1.0874015125327023, tolerance = 1e-14)
})

test_that("ESS-derived statistics match at short split-chain boundaries", {
  stats <- c(
    "ess_bulk",
    "ess_tail",
    "ess_basic",
    "ess_mean",
    "ess_sd",
    "mcse_mean",
    "mcse_sd"
  )

  for (niter in 6:12) {
    x <- make_draws(niter, 4, 3, seed = 100 + niter)
    suppressWarnings(expect_matches_direct_references(x, stats = stats))
  }
})

test_that("rank-normalised diagnostics retain distinct sub-epsilon values", {
  eps <- .Machine$double.eps
  values <- rep(c(0, eps / 4, eps / 2, 3 * eps / 4), length.out = 200)
  x <- posterior::as_draws_array(array(values, dim = c(50, 4, 1)))

  out <- posteriorcpp::summarise_draws_cpp(
    x,
    stats = c("rhat", "rhat_basic", "ess_bulk", "ess_basic")
  )
  m <- posterior::extract_variable_matrix(x, "...1")

  expect_equal(out$rhat, posterior::rhat(m), tolerance = 1e-14)
  expect_equal(out$ess_bulk, posterior::ess_bulk(m), tolerance = 1e-4)
  expect_true(is.na(out$rhat_basic))
  expect_true(is.na(out$ess_basic))
})

test_that("non-finite draws follow direct reference semantics", {
  set.seed(2026)
  arr <- array(rnorm(80 * 4), dim = c(80, 4, 4))
  arr[1, 1, 2] <- Inf
  arr[1, 1, 3] <- -Inf
  arr[1, 1, 4] <- -Inf
  arr[2, 1, 4] <- Inf
  x <- posterior::as_draws_array(arr)

  out <- expect_matches_direct_references(x)
  expect_true(all(is.na(out$ess_tail[2:4])))
  expect_false(is.na(out$ess_tail[[1]]))
})

test_that("scalar summaries match on deterministic ties and extreme values", {
  xmax <- .Machine$double.xmax
  tied <- rep(c(-3, -3, -1, 0, 0, 0, 2, 2, 5, 5), 8)
  extreme_q5 <- c(-xmax, rep(xmax, 19))
  extreme_q95 <- c(rep(-xmax, 19), xmax)
  arr <- array(0, dim = c(20, 4, 3))
  arr[, , 1] <- matrix(tied, nrow = 20)
  arr[, , 2] <- matrix(rep(extreme_q5, 4), nrow = 20)
  arr[, , 3] <- matrix(rep(extreme_q95, 4), nrow = 20)
  x <- posterior::as_draws_array(arr)

  expect_matches_direct_references(
    x,
    stats = c("mean", "median", "sd", "var", "mad", "q5", "q95")
  )
})

test_that("a single draw preserves NA rather than NaN variance semantics", {
  x <- posterior::as_draws_array(array(3, dim = c(1, 1, 1)))
  out <- posteriorcpp::summarise_draws_cpp(x, stats = c("sd", "var"))

  expect_identical(out$sd, sd(3))
  expect_identical(out$var, var(3))
  expect_false(is.nan(out$sd))
  expect_false(is.nan(out$var))
})

test_that("large same-sign finite values do not overflow the mean", {
  xmax <- .Machine$double.xmax
  x <- posterior::as_draws_array(array(rep(xmax, 12), dim = c(3, 4, 1)))
  out <- posteriorcpp::summarise_draws_cpp(x, stats = c("mean", "sd", "var"))

  expect_identical(out$mean, xmax)
  expect_identical(out$sd, 0)
  expect_identical(out$var, 0)
})

test_that("all supported statistics match across difficult finite draws", {
  distributions <- list(
    skewed = function(n) rexp(n, rate = 0.2),
    heavy_tailed = function(n) rt(n, df = 3),
    discrete = function(n) sample(c(-2, -1, 0, 0, 0, 3), n, replace = TRUE)
  )

  for (i in seq_along(distributions)) {
    x <- make_draws(500, 4, 5, seed = 700 + i, fun = distributions[[i]])
    expect_matches_direct_references(x)
  }
})
