test_that("NA/NaN draws for a variable propagate as NA across all stats for that variable only", {
  set.seed(1)
  arr <- array(rnorm(200 * 4 * 3), dim = c(200, 4, 3))
  arr[5, 2, 2] <- NaN
  x <- posterior::as_draws_array(arr)

  out <- posteriorcpp::summarise_draws(x)
  expect_true(all(is.na(out[out$variable == "...2", all_stats])))
  expect_true(all(!is.na(out[out$variable == "...1", all_stats])))
  expect_true(all(!is.na(out[out$variable == "...3", all_stats])))

  expect_matches_posterior(x)
})

test_that("multiple variables with NA are each handled independently", {
  set.seed(2)
  arr <- array(rnorm(300 * 4 * 5), dim = c(300, 4, 5))
  arr[1, 1, 1] <- NA_real_
  arr[10, 3, 4] <- NaN
  x <- posterior::as_draws_array(arr)
  expect_matches_posterior(x)
})

test_that("a constant (zero-variance) variable yields NA convergence diagnostics", {
  set.seed(3)
  arr <- array(rnorm(200 * 4 * 2), dim = c(200, 4, 2))
  arr[, , 1] <- 7 # constant across all draws
  x <- posterior::as_draws_array(arr)

  out <- posteriorcpp::summarise_draws(x)
  const_row <- out[out$variable == "...1", ]
  expect_equal(const_row$mean, 7)
  expect_equal(const_row$median, 7)
  expect_equal(const_row$sd, 0)
  expect_equal(const_row$mad, 0)
  expect_true(is.na(const_row$rhat))
  expect_true(is.na(const_row$ess_bulk))
  expect_true(is.na(const_row$ess_tail))

  expect_matches_posterior(x)
})

test_that("single-chain draws (nchains = 1) are handled consistently", {
  x <- make_draws(500, 1, 4, seed = 4)
  expect_matches_posterior(x)
})

test_that("very short chains (niter = 1, 3) do not error and match posterior", {
  # niter = 2 is excluded here — see the dedicated test below.
  for (niter in c(1, 3)) {
    x <- make_draws(niter, 4, 3, seed = niter + 10)
    expect_no_error(out <- posteriorcpp::summarise_draws(x))
    expect_equal(nrow(out), 3)
    expect_matches_posterior(x)
  }
})

test_that("niter = 2 does not error (known divergence from posterior's ess_tail)", {
  # posterior's .split_chains() drops a single-row half to a vector, so its
  # ess_tail can escape the niter<3 guard; posteriorcpp's explicit split
  # returns NA. A quirk of posterior, not a bug to replicate.
  x <- make_draws(2, 4, 3, seed = 12)
  expect_no_error(out <- posteriorcpp::summarise_draws(x))
  expect_equal(nrow(out), 3)
  expect_true(all(is.na(out$rhat)))
  expect_true(all(is.na(out$ess_bulk)))
  expect_true(all(is.na(out$ess_tail)))
})

test_that("a single draw total (niter = 1, nchains = 1) does not error", {
  x <- make_draws(1, 1, 2, seed = 11)
  expect_no_error(out <- posteriorcpp::summarise_draws(x))
  expect_equal(nrow(out), 2)
})

test_that("large-magnitude and near-zero-variance draws do not break rank normalisation", {
  set.seed(12)
  arr <- array(rnorm(300 * 4 * 2, mean = 0, sd = 1), dim = c(300, 4, 2))
  arr[, , 1] <- arr[, , 1] * 1e8 + 1e10       # very large magnitude
  arr[, , 2] <- arr[, , 2] * 1e-8              # near-constant / tiny variance
  x <- posterior::as_draws_array(arr)
  expect_matches_posterior(x)
})

test_that("many tied values (heavy ranking ties) match posterior's average-rank handling", {
  set.seed(13)
  arr <- array(sample(1:5, 400 * 4 * 3, replace = TRUE) + 0, dim = c(400, 4, 3))
  storage.mode(arr) <- "double"
  x <- posterior::as_draws_array(arr)
  expect_matches_posterior(x)
})
