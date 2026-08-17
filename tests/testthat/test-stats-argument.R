test_that("each individual stat, requested alone, matches posterior", {
  x <- make_draws(600, 4, 6, seed = 1)
  for (s in all_stats) {
    expect_matches_posterior(x, stats = s)
  }
})

test_that("common stat subsets match posterior and preserve canonical column order", {
  x <- make_draws(600, 4, 6, seed = 2)
  combos <- list(
    c("mean", "sd"),
    c("median", "mad"),
    c("q5", "q95"),
    c("rhat", "ess_bulk"),
    c("rhat", "ess_tail"),
    c("ess_bulk", "ess_tail"),
    c("mean", "median", "sd", "mad", "q5", "q95"),
    all_stats
  )
  for (combo in combos) {
    expect_matches_posterior(x, stats = combo)
  }
})

test_that("stats order in the call does not affect output column order", {
  x <- make_draws(300, 4, 4, seed = 3)
  out <- posteriorcpp::summarise_draws(x, stats = c("ess_tail", "mean", "rhat", "sd"))
  expect_identical(names(out), c("variable", "mean", "sd", "rhat", "ess_tail"))
})

test_that("duplicate stats are collapsed to a single column", {
  x <- make_draws(300, 4, 3, seed = 4)
  out <- posteriorcpp::summarise_draws(x, stats = c("mean", "mean", "sd", "sd", "sd"))
  expect_identical(names(out), c("variable", "mean", "sd"))
})

test_that("requesting all stats individually reconstructs the default output", {
  x <- make_draws(400, 4, 5, seed = 5)
  full <- posteriorcpp::summarise_draws(x)
  piecewise <- lapply(all_stats, function(s) posteriorcpp::summarise_draws(x, stats = s)[[s]])
  names(piecewise) <- all_stats
  for (s in all_stats) {
    expect_equal(piecewise[[s]], full[[s]])
  }
})

test_that("an invalid stat name errors with a helpful message", {
  x <- make_draws(200, 4, 2, seed = 6)
  expect_error(posteriorcpp::summarise_draws(x, stats = "bogus"))
  expect_error(posteriorcpp::summarise_draws(x, stats = c("mean", "not_a_stat")))
})

test_that("an empty stats vector errors rather than silently returning nothing", {
  x <- make_draws(200, 4, 2, seed = 7)
  expect_error(posteriorcpp::summarise_draws(x, stats = character(0)))
})

test_that("stats subsetting still handles NaN draws correctly", {
  set.seed(8)
  arr <- array(rnorm(200 * 4 * 2), dim = c(200, 4, 2))
  arr[1, 1, 1] <- NaN
  x <- posterior::as_draws_array(arr)

  out <- posteriorcpp::summarise_draws(x, stats = c("mean", "rhat"))
  expect_true(all(is.na(out[out$variable == "...1", c("mean", "rhat")])))
  expect_true(all(!is.na(out[out$variable == "...2", c("mean", "rhat")])))
})

test_that("shared-work stats (rhat + ess_bulk) still match when requested together or separately", {
  x <- make_draws(500, 4, 8, seed = 9)
  together <- posteriorcpp::summarise_draws(x, stats = c("rhat", "ess_bulk"))
  rhat_only <- posteriorcpp::summarise_draws(x, stats = "rhat")
  ess_only <- posteriorcpp::summarise_draws(x, stats = "ess_bulk")

  expect_equal(together$rhat, rhat_only$rhat)
  expect_equal(together$ess_bulk, ess_only$ess_bulk)
})
