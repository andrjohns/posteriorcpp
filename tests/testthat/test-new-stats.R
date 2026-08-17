# Tests for the opt-in stats added beyond posterior::summarise_draws()'s own
# default set: "var", "rhat_basic", "ess_basic". Reference values come
# directly from the matching posterior functions, not from
# posterior::summarise_draws() itself, since these aren't in its default
# funs list.
#
# Note on "var": summarise_draws()/sd() pass each variable through as a
# flattened (niter x nchains) scalar (sd() explicitly flattens non-vector
# input), whereas posterior::variance()/var() called directly on a bare
# matrix instead give per-chain variances or a covariance matrix. The
# correct scalar reference is therefore stats::var(as.vector(m)), matching
# how sd() (and thus posteriorcpp's existing "sd" column) already treats the
# matrix.

reference_value <- function(stat, m) {
  switch(stat,
    var = stats::var(as.vector(m)),
    rhat_basic = posterior::rhat_basic(m),
    ess_basic = posterior::ess_basic(m)
  )
}

expect_matches_reference <- function(x, stat, tolerance = 1e-4) {
  out <- posteriorcpp::summarise_draws(x, stats = stat)
  vars <- posterior::variables(x)
  expected <- vapply(vars, function(v) {
    reference_value(stat, posterior::extract_variable_matrix(x, v))
  }, numeric(1))
  out <- out[order(out$variable), ]
  expected <- expected[order(names(expected))]
  expect_equal(out[[stat]], unname(expected), tolerance = tolerance)
  invisible(out)
}

test_that("default stats are unchanged by adding new opt-in stats", {
  x <- make_draws(300, 4, 5, seed = 1)
  out <- posteriorcpp::summarise_draws(x)
  expect_identical(names(out), c("variable", all_stats))
  expect_matches_posterior(x)
})

test_that("var matches stats::var(as.vector(m)) and equals sd^2", {
  x <- make_draws(700, 4, 10, seed = 2)
  expect_matches_reference(x, "var")

  out <- posteriorcpp::summarise_draws(x, stats = c("sd", "var"))
  expect_equal(out$var, out$sd^2, tolerance = 1e-8)
})

test_that("rhat_basic matches posterior::rhat_basic()", {
  x <- make_draws(700, 4, 10, seed = 3)
  expect_matches_reference(x, "rhat_basic")
})

test_that("ess_basic matches posterior::ess_basic()", {
  x <- make_draws(700, 4, 10, seed = 4)
  expect_matches_reference(x, "ess_basic")
})

test_that("rhat_basic differs from rank-normalised rhat on skewed data", {
  # rhat_basic omits the rank-normalisation step, so on a distribution with
  # heavy skew the two diagnostics need not agree even on well-mixed chains.
  x <- make_draws(1000, 4, 5, seed = 5, fun = function(n) stats::rexp(n, rate = 0.5))
  out <- posteriorcpp::summarise_draws(x, stats = c("rhat", "rhat_basic"))
  expect_false(isTRUE(all.equal(out$rhat, out$rhat_basic)))
  expect_matches_reference(x, "rhat_basic")
})

test_that("requesting rhat_basic and rhat together matches requesting them separately", {
  x <- make_draws(600, 4, 8, seed = 6)
  together <- posteriorcpp::summarise_draws(x, stats = c("rhat", "rhat_basic"))
  rhat_only <- posteriorcpp::summarise_draws(x, stats = "rhat")
  rb_only <- posteriorcpp::summarise_draws(x, stats = "rhat_basic")
  expect_equal(together$rhat, rhat_only$rhat)
  expect_equal(together$rhat_basic, rb_only$rhat_basic)
})

test_that("requesting ess_basic and ess_bulk together matches requesting them separately", {
  x <- make_draws(600, 4, 8, seed = 7)
  together <- posteriorcpp::summarise_draws(x, stats = c("ess_bulk", "ess_basic"))
  bulk_only <- posteriorcpp::summarise_draws(x, stats = "ess_bulk")
  basic_only <- posteriorcpp::summarise_draws(x, stats = "ess_basic")
  expect_equal(together$ess_bulk, bulk_only$ess_bulk)
  expect_equal(together$ess_basic, basic_only$ess_basic)
})

test_that("all new stats requested together, in reverse order, still match individually", {
  x <- make_draws(500, 4, 6, seed = 8)
  out <- posteriorcpp::summarise_draws(x, stats = c("ess_basic", "rhat_basic", "var"))
  expect_identical(names(out), c("variable", "var", "rhat_basic", "ess_basic"))
  for (s in c("var", "rhat_basic", "ess_basic")) {
    expect_matches_reference(x, s)
  }
})

test_that("full stat set is in the documented canonical column order", {
  x <- make_draws(300, 4, 4, seed = 9)
  out <- posteriorcpp::summarise_draws(x, stats = posteriorcpp:::.posteriorcpp_all_stats)
  expect_identical(
    names(out),
    c("variable", "mean", "median", "sd", "var", "mad", "q5", "q95",
      "rhat", "rhat_basic", "ess_bulk", "ess_tail", "ess_basic",
      "ess_mean", "ess_sd", "mcse_mean", "mcse_sd")
  )
})

test_that("new stats propagate NA on NaN draws, independently per variable", {
  set.seed(10)
  arr <- array(rnorm(200 * 4 * 3), dim = c(200, 4, 3))
  arr[5, 2, 2] <- NaN
  x <- posterior::as_draws_array(arr)

  out <- posteriorcpp::summarise_draws(x, stats = c("var", "rhat_basic", "ess_basic"))
  expect_true(all(is.na(out[out$variable == "...2", c("var", "rhat_basic", "ess_basic")])))
  expect_true(all(!is.na(out[out$variable == "...1", c("var", "rhat_basic", "ess_basic")])))
  expect_true(all(!is.na(out[out$variable == "...3", c("var", "rhat_basic", "ess_basic")])))
})

test_that("a constant (zero-variance) variable yields var = 0 and NA rhat_basic/ess_basic", {
  set.seed(11)
  arr <- array(rnorm(200 * 4 * 2), dim = c(200, 4, 2))
  arr[, , 1] <- 3
  x <- posterior::as_draws_array(arr)

  out <- posteriorcpp::summarise_draws(x, stats = c("var", "rhat_basic", "ess_basic"))
  const_row <- out[out$variable == "...1", ]
  expect_equal(const_row$var, 0)
  expect_true(is.na(const_row$rhat_basic))
  expect_true(is.na(const_row$ess_basic))
})

test_that("an unknown stat still errors with the full stat choice list", {
  x <- make_draws(200, 4, 2, seed = 12)
  err <- tryCatch(posteriorcpp::summarise_draws(x, stats = "bogus"), error = function(e) e)
  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "var", fixed = TRUE)
  expect_match(conditionMessage(err), "rhat_basic", fixed = TRUE)
  expect_match(conditionMessage(err), "ess_basic", fixed = TRUE)
})
