test_that("matches loo::loo() across a sweep of shapes with auto r_eff", {
  shapes <- list(
    c(niter = 200, nchains = 4, nobs = 10),
    c(niter = 1000, nchains = 4, nobs = 50),
    c(niter = 2000, nchains = 2, nobs = 30),
    c(niter = 300, nchains = 1, nobs = 20), # single chain
    c(niter = 4000, nchains = 4, nobs = 5)
  )
  for (i in seq_along(shapes)) {
    s <- shapes[[i]]
    x <- make_log_lik(s[["niter"]], s[["nchains"]], s[["nobs"]], seed = 300 + i)
    expect_matches_loo(x)
  }
})

test_that("matches loo::loo() with r_eff = FALSE (r_eff = 1 for every observation)", {
  x <- make_log_lik(500, 4, 20, seed = 42)
  expect_matches_loo(x, r_eff = FALSE)
})

test_that("matches loo::loo() with a user-supplied scalar r_eff", {
  x <- make_log_lik(500, 4, 20, seed = 43)
  expect_matches_loo(x, r_eff = 0.7)
})

test_that("matches loo::loo() with a user-supplied per-observation r_eff vector", {
  x <- make_log_lik(500, 4, 20, seed = 44)
  r_eff <- runif(20, 0.3, 1)
  expect_matches_loo(x, r_eff = r_eff)
})

test_that("auto r_eff matches loo::loo() even for very-negative log-lik columns", {
  # loo::relative_eff(exp(x)) collapses to NA for these via absolute-scale
  # underflow in the degeneracy check; loo_cpp()'s internal shift avoids it.
  niter <- 1000
  nchains <- 4
  nobs <- 20
  set.seed(45)
  shift <- matrix(
    rep(seq_len(nobs) * 5, each = niter * nchains),
    niter * nchains,
    nobs
  )
  x <- array(
    rnorm(niter * nchains * nobs, sd = 2) - shift,
    dim = c(niter, nchains, nobs)
  )
  expect_matches_loo(x)
})

test_that("returns the loo package's object structure", {
  x <- make_log_lik(500, 4, 10, seed = 5)
  out <- suppressWarnings(loo_cpp(x))

  expect_s3_class(out, c("psis_loo", "importance_sampling_loo", "loo"))
  expect_named(
    out,
    c(
      "estimates",
      "pointwise",
      "diagnostics",
      "psis_object",
      "elpd_loo",
      "p_loo",
      "looic",
      "se_elpd_loo",
      "se_p_loo",
      "se_looic"
    )
  )
  expect_equal(dimnames(out$estimates)[[1]], c("elpd_loo", "p_loo", "looic"))
  expect_equal(dimnames(out$estimates)[[2]], c("Estimate", "SE"))
  expect_equal(
    colnames(out$pointwise),
    c("elpd_loo", "mcse_elpd_loo", "p_loo", "looic", "influence_pareto_k")
  )
  expect_named(out$diagnostics, c("pareto_k", "n_eff", "r_eff"))
  expect_null(out$psis_object)
  expect_equal(dim(out), c(2000L, 10L))
})

test_that("save_psis = TRUE attaches a usable psis_object", {
  x <- make_log_lik(500, 4, 10, seed = 6)
  out <- suppressWarnings(loo_cpp(x, save_psis = TRUE))

  expect_true(loo::is.psis(out$psis_object))
  expect_equal(dim(out$psis_object), c(2000L, 10L))
  w <- weights(out$psis_object, normalize = TRUE, log = FALSE)
  expect_equal(colSums(w), rep(1, 10), tolerance = 1e-8)
})

test_that("interoperates with loo::loo_compare()", {
  x1 <- make_log_lik(500, 4, 15, seed = 7)
  x2 <- x1 + rnorm(length(x1), sd = 0.1)
  l1 <- suppressWarnings(loo_cpp(x1))
  l2 <- suppressWarnings(loo_cpp(x2))
  comp <- loo::loo_compare(list(model1 = l1, model2 = l2))
  expect_setequal(comp$model, c("model1", "model2"))
})

test_that("warns about high Pareto k values", {
  # A single wild outlier draw in one chain produces an unreliable tail fit.
  set.seed(8)
  x <- make_log_lik(200, 4, 5, sd = 1)
  x[1, 1, 1] <- -200
  expect_warning(loo_cpp(x), "Pareto k")
})

test_that("`r_eff` argument validates its length", {
  x <- make_log_lik(200, 4, 5, seed = 9)
  expect_error(loo_cpp(x, r_eff = c(1, 2)), "one value per observation")
})

test_that("`x` must be a 3-D array", {
  expect_error(loo_cpp(matrix(1, 2, 2)), "3-D")
})

test_that("NAs are rejected", {
  x <- make_log_lik(200, 4, 5, seed = 10)
  x[1, 1, 1] <- NA
  expect_error(loo_cpp(x), "NA")
})
