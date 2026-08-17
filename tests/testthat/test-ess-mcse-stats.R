# Reference values come from the matching posterior functions directly, not
# from posterior::summarise_draws().

reference_value2 <- function(stat, m) {
  switch(stat,
    ess_mean = posterior::ess_mean(m),
    ess_sd = posterior::ess_sd(m),
    mcse_mean = posterior::mcse_mean(m),
    mcse_sd = posterior::mcse_sd(m)
  )
}

expect_matches_reference2 <- function(x, stat, tolerance = 1e-4) {
  out <- posteriorcpp::summarise_draws(x, stats = stat)
  vars <- posterior::variables(x)
  expected <- vapply(vars, function(v) {
    reference_value2(stat, posterior::extract_variable_matrix(x, v))
  }, numeric(1))
  out <- out[order(out$variable), ]
  expected <- expected[order(names(expected))]
  expect_equal(out[[stat]], unname(expected), tolerance = tolerance)
  invisible(out)
}

test_that("ess_mean matches posterior::ess_mean() and equals ess_basic exactly", {
  x <- make_draws(700, 4, 10, seed = 1)
  expect_matches_reference2(x, "ess_mean")

  out <- posteriorcpp::summarise_draws(x, stats = c("ess_basic", "ess_mean"))
  expect_identical(out$ess_basic, out$ess_mean)
})

test_that("ess_sd matches posterior::ess_sd()", {
  x <- make_draws(700, 4, 10, seed = 2)
  expect_matches_reference2(x, "ess_sd")
})

test_that("mcse_mean matches posterior::mcse_mean()", {
  x <- make_draws(700, 4, 10, seed = 3)
  expect_matches_reference2(x, "mcse_mean")
})

test_that("mcse_sd matches posterior::mcse_sd()", {
  x <- make_draws(700, 4, 10, seed = 4)
  expect_matches_reference2(x, "mcse_sd")
})

test_that("mcse_mean requested alone matches when sd/ess_basic/ess_mean are also requested", {
  x <- make_draws(500, 4, 6, seed = 5)
  alone <- posteriorcpp::summarise_draws(x, stats = "mcse_mean")
  together <- posteriorcpp::summarise_draws(x, stats = c("mcse_mean", "sd", "ess_basic", "ess_mean"))
  expect_equal(alone$mcse_mean, together$mcse_mean)
})

test_that("mcse_sd requested alone matches when ess_sd is also requested", {
  x <- make_draws(500, 4, 6, seed = 6)
  alone <- posteriorcpp::summarise_draws(x, stats = "mcse_sd")
  together <- posteriorcpp::summarise_draws(x, stats = c("mcse_sd", "ess_sd"))
  expect_equal(alone$mcse_sd, together$mcse_sd)
})

test_that("ess_sd and ess_mean are generally different (distinct estimators)", {
  x <- make_draws(800, 4, 5, seed = 7, fun = function(n) stats::rt(n, df = 3))
  out <- posteriorcpp::summarise_draws(x, stats = c("ess_mean", "ess_sd"))
  expect_false(isTRUE(all.equal(out$ess_mean, out$ess_sd)))
})

test_that("all four stats requested together, unordered, still match individually", {
  x <- make_draws(600, 4, 6, seed = 8)
  out <- posteriorcpp::summarise_draws(
    x, stats = c("mcse_sd", "ess_mean", "mcse_mean", "ess_sd")
  )
  expect_identical(names(out), c("variable", "ess_mean", "ess_sd", "mcse_mean", "mcse_sd"))
  for (s in c("ess_mean", "ess_sd", "mcse_mean", "mcse_sd")) {
    expect_matches_reference2(x, s)
  }
})

test_that("default stats are still unaffected by this batch", {
  x <- make_draws(300, 4, 5, seed = 9)
  expect_matches_posterior(x)
})

test_that("new stats propagate NA on NaN draws, independently per variable", {
  set.seed(10)
  arr <- array(rnorm(200 * 4 * 3), dim = c(200, 4, 3))
  arr[5, 2, 2] <- NaN
  x <- posterior::as_draws_array(arr)

  stats4 <- c("ess_mean", "ess_sd", "mcse_mean", "mcse_sd")
  out <- posteriorcpp::summarise_draws(x, stats = stats4)
  expect_true(all(is.na(out[out$variable == "...2", stats4])))
  expect_true(all(!is.na(out[out$variable == "...1", stats4])))
  expect_true(all(!is.na(out[out$variable == "...3", stats4])))
})

test_that("a constant (zero-variance) variable yields NA for all four stats", {
  set.seed(11)
  arr <- array(rnorm(200 * 4 * 2), dim = c(200, 4, 2))
  arr[, , 1] <- 4
  x <- posterior::as_draws_array(arr)

  stats4 <- c("ess_mean", "ess_sd", "mcse_mean", "mcse_sd")
  out <- posteriorcpp::summarise_draws(x, stats = stats4)
  const_row <- out[out$variable == "...1", ]
  expect_true(all(is.na(const_row[stats4])))
})

test_that("full stat set request includes all four in canonical position", {
  x <- make_draws(300, 4, 3, seed = 12)
  out <- posteriorcpp::summarise_draws(x, stats = posteriorcpp:::.posteriorcpp_all_stats)
  tail4 <- utils::tail(names(out), 4)
  expect_identical(tail4, c("ess_mean", "ess_sd", "mcse_mean", "mcse_sd"))
})
