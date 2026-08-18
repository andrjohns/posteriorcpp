# Numerical accuracy/consistency for the inputs most likely to break a
# hand-rolled PSIS-LOO implementation: degenerate columns, tiny sample sizes,
# heavy tails, extreme r_eff, boundary tail lengths, and scale extremes.
# Each case is checked against loo::loo() to a tight tolerance, not just for
# "doesn't crash."

test_that("matches loo::loo() for an entirely-constant (zero-variance) column", {
  set.seed(101)
  x <- make_log_lik(400, 4, 4, seed = 101)
  x[,, 2] <- 5 # constant log_lik for this observation
  expect_matches_loo(x)
})

test_that("matches loo::loo() when every column is constant", {
  x <- array(3.14, dim = c(200, 4, 3))
  expect_matches_loo(x)
})

test_that("matches loo::loo() for a column with only a handful of distinct values", {
  # Discrete-outcome log-lik (e.g. small-count Poisson/binomial) produces
  # heavy ties in the PSIS tail, which can force the GPD grid estimator's
  # xstar <= min(x) branch (k_hat = NA -> Inf sentinel).
  set.seed(102)
  x <- make_log_lik(600, 4, 6, seed = 102)
  x[,, 1] <- round(x[,, 1], 1) # collapse to ~20 distinct values
  x[,, 2] <- round(x[,, 2] * 2) / 2 # even coarser rounding

  # log_weights is skipped: which of several *exactly tied* raw log-lik
  # draws gets which PSIS-smoothed quantile is genuinely unspecified by the
  # algorithm (R's sort.int() and std::sort() break ties differently). That
  # doesn't affect any statistic that matters -- elpd_loo/p_loo/looic/mcse
  # sum over draws with identical log_lik regardless of which tied draw got
  # which weight -- so those are still checked (and must match exactly) via
  # expect_matches_loo(); this only confirms the weight *values* used
  # per column are the same multiset.
  mine <- expect_matches_loo(x, check_log_weights = FALSE)
  ref_r_eff <- shifted_relative_eff(x)
  ref <- suppressWarnings(loo::loo(x, r_eff = ref_r_eff, save_psis = TRUE))
  for (col in seq_len(dim(x)[3])) {
    expect_equal(
      sort(mine$psis_object$log_weights[, col]),
      sort(ref$psis_object$log_weights[, col]),
      tolerance = 1e-8
    )
  }
})

test_that("matches loo::loo() with heavy-tailed (Cauchy) log-lik", {
  set.seed(103)
  x <- array(rcauchy(300 * 4 * 5), dim = c(300, 4, 5))
  expect_matches_loo(x)
})

test_that("matches loo::loo() with a single extreme outlier draw", {
  set.seed(104)
  x <- make_log_lik(300, 4, 8, seed = 104)
  x[1, 1, 1] <- x[1, 1, 1] - 50
  x[5, 3, 4] <- x[5, 3, 4] - 200
  expect_matches_loo(x)
})

test_that("matches loo::loo() at the minimum viable niter (niter = 2)", {
  # r_eff is fixed rather than auto here: posterior's own .split_chains()
  # has a dimension-drop quirk for niter in {2, 3} (selecting a single row
  # from a matrix collapses it to a vector, so cbind() transposes the split
  # instead of concatenating it) that makes its ESS estimate for such tiny
  # niter an artifact of that quirk rather than a meaningful statistic. This
  # test instead isolates and checks the core PSIS-LOO math at this boundary.
  x <- make_log_lik(2, 4, 3, seed = 105)
  expect_matches_loo(x, r_eff = 1)
})

test_that("auto r_eff falls back to 1 (not NA/garbage) for niter in {2, 3}", {
  for (niter in c(2, 3)) {
    x <- make_log_lik(niter, 4, 3, seed = 2100 + niter)
    out <- suppressWarnings(loo_cpp(x, r_eff = TRUE))
    expect_true(all(is.finite(out$diagnostics$r_eff)))
    expect_equal(out$diagnostics$r_eff, rep(1, 3))
  }
})

test_that("matches loo::loo() across small niter values spanning split_chains parity", {
  # r_eff fixed for the same reason as the niter=2 test above; niter=3 hits
  # the same .split_chains() quirk as niter=2.
  for (niter in c(3, 4, 5, 6, 7)) {
    x <- make_log_lik(niter, 4, 4, seed = 1000 + niter)
    expect_matches_loo(x, r_eff = 1)
  }
})

test_that("matches loo::loo() around the tail-length enough-samples boundary", {
  # tail_len = ceiling(min(0.2*S, 3*sqrt(S))); S = niter*nchains chosen so
  # tail_len sits on both sides of the enough_tail_samples() cutoff of 5.
  for (niter in c(5, 6, 8, 25)) {
    x <- make_log_lik(niter, 4, 6, seed = 2000 + niter)
    expect_matches_loo(x)
  }
})

test_that("matches loo::loo() with a single chain", {
  x <- make_log_lik(500, 1, 10, seed = 106)
  expect_matches_loo(x)
})

test_that("matches loo::loo() with a single observation", {
  x <- make_log_lik(500, 4, 1, seed = 107)
  expect_matches_loo(x)
})

test_that("matches loo::loo() with very large-magnitude r_eff", {
  x <- make_log_lik(500, 4, 10, seed = 108)
  expect_matches_loo(x, r_eff = 100)
})

test_that("matches loo::loo() with very small r_eff (tail length capped at 20%)", {
  x <- make_log_lik(500, 4, 10, seed = 109)
  expect_matches_loo(x, r_eff = 0.001)
})

test_that("matches loo::loo() with mixed per-observation r_eff spanning extremes", {
  x <- make_log_lik(500, 4, 10, seed = 110)
  r_eff <- c(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 20, 100, 0.3)
  expect_matches_loo(x, r_eff = r_eff)
})

test_that("matches loo::loo() for very large-magnitude log-lik values", {
  set.seed(111)
  x <- make_log_lik(500, 4, 6, seed = 111, sd = 1) * 1000 - 5000
  expect_matches_loo(x)
})

test_that("matches loo::loo() for near-constant columns just above the degeneracy threshold", {
  set.seed(112)
  x <- make_log_lik(400, 4, 4, seed = 112)
  # Perturb one column by an amount comfortably above double-precision EPS
  # but far below anything visually distinguishable from constant.
  x[,, 3] <- 5 + (x[,, 3] - mean(x[,, 3])) * 1e-8
  expect_matches_loo(x)
})

test_that("matches loo::loo() when chains disagree strongly (poor mixing)", {
  set.seed(113)
  niter <- 400
  nchains <- 4
  nobs <- 5
  x <- array(rnorm(niter * nchains * nobs), dim = c(niter, nchains, nobs))
  for (c in seq_len(nchains)) {
    x[, c, ] <- x[, c, ] + c * 8
  }
  expect_matches_loo(x)
})

test_that("matches loo::loo() for autocorrelated (slow-mixing) draws", {
  set.seed(114)
  niter <- 800
  nchains <- 4
  nobs <- 3
  x <- array(0, dim = c(niter, nchains, nobs))
  for (c in seq_len(nchains)) {
    for (v in seq_len(nobs)) {
      innov <- rnorm(niter)
      chain <- numeric(niter)
      chain[1] <- innov[1]
      for (t in 2:niter) {
        chain[t] <- 0.97 * chain[t - 1] + innov[t]
      }
      x[, c, v] <- chain
    }
  }
  expect_matches_loo(x)
})

test_that("matches loo::loo() for a large number of observations (TBB parallel path)", {
  x <- make_log_lik(300, 4, 400, seed = 115)
  expect_matches_loo(x)
})

test_that("matches loo::example_loglik_array(), the loo package's own reference fixture", {
  x <- loo::example_loglik_array()
  expect_matches_loo(x)
})

test_that("results are deterministic across repeated calls (parallel-safety)", {
  x <- make_log_lik(300, 4, 200, seed = 116)
  out1 <- suppressWarnings(loo_cpp(x, save_psis = TRUE))
  out2 <- suppressWarnings(loo_cpp(x, save_psis = TRUE))
  expect_identical(out1$estimates, out2$estimates)
  expect_identical(out1$pointwise, out2$pointwise)
  expect_identical(out1$diagnostics, out2$diagnostics)
  expect_identical(out1$psis_object$log_weights, out2$psis_object$log_weights)
})

test_that("+Inf log-lik degrades gracefully (no error, no crash) like loo::loo()", {
  x <- make_log_lik(300, 4, 4, seed = 117)
  x[1, 1, 1] <- Inf
  ref <- suppressWarnings(loo::loo(x, r_eff = 1))
  mine <- suppressWarnings(loo_cpp(x, r_eff = 1))
  # The +Inf-affected observation is unusable in both implementations...
  expect_true(is.na(ref$pointwise[1, "elpd_loo"]))
  expect_true(is.na(mine$pointwise[1, "elpd_loo"]))
  # ...but the unaffected observations must still match exactly.
  expect_equal(
    mine$pointwise[-1, "elpd_loo"],
    ref$pointwise[-1, "elpd_loo"],
    tolerance = 1e-8
  )
})

test_that("-Inf log-lik is rejected with a clear error, not silently corrupted", {
  x <- make_log_lik(300, 4, 4, seed = 118)
  x[1, 1, 1] <- -Inf
  expect_error(loo_cpp(x), "-Inf")
})

test_that("r_eff vector containing NA is replaced with 1 (matches loo::loo())", {
  x <- make_log_lik(300, 4, 4, seed = 119)
  r_eff <- c(1, NA, 0.5, NA)
  ref <- suppressWarnings(loo::loo(x, r_eff = r_eff))
  mine <- suppressMessages(suppressWarnings(loo_cpp(x, r_eff = r_eff)))
  expect_equal(mine$diagnostics$r_eff, ref$diagnostics$r_eff, tolerance = 1e-8)
  expect_equal(
    unclass(mine$estimates),
    unclass(ref$estimates),
    tolerance = 1e-8
  )
})

test_that("r_eff = NA (scalar) is replaced with 1 for every observation", {
  x <- make_log_lik(300, 4, 4, seed = 120)
  expect_message(
    out <- loo_cpp(x, r_eff = NA),
    "Replacing NAs"
  )
  expect_equal(out$diagnostics$r_eff, rep(1, 4))
})
