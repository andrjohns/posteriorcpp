# Numerical consistency with posterior::summarise_draws() across a broad
# sweep of draws shapes and underlying distributions.

test_that("matches posterior across a sweep of niter/nchains/nvars shapes", {
  shapes <- list(
    c(niter = 100,  nchains = 2,  nvars = 1),
    c(niter = 100,  nchains = 4,  nvars = 8),
    c(niter = 500,  nchains = 4,  nvars = 20),
    c(niter = 1000, nchains = 4,  nvars = 5),
    c(niter = 2000, nchains = 8,  nvars = 3),
    c(niter = 300,  nchains = 1,  nvars = 6),   # single chain
    c(niter = 4000, nchains = 2,  nvars = 2)
  )

  for (i in seq_along(shapes)) {
    s <- shapes[[i]]
    x <- make_draws(s[["niter"]], s[["nchains"]], s[["nvars"]], seed = 100 + i)
    expect_matches_posterior(x)
  }
})

test_that("matches posterior for odd niter (uneven split_chains halves)", {
  for (niter in c(51, 99, 151, 999, 1999)) {
    x <- make_draws(niter, 4, 4, seed = niter)
    expect_matches_posterior(x)
  }
})

test_that("matches posterior across non-normal underlying distributions", {
  distributions <- list(
    uniform    = function(n) stats::runif(n, -5, 5),
    exponential = function(n) stats::rexp(n, rate = 2),
    student_t  = function(n) stats::rt(n, df = 3),
    bimodal    = function(n) stats::rnorm(n) + sample(c(-4, 4), n, replace = TRUE),
    heavy_tail = function(n) stats::rcauchy(n)
  )

  for (nm in names(distributions)) {
    x <- make_draws(500, 4, 6, seed = 200, fun = distributions[[nm]])
    expect_matches_posterior(x)
  }
})

test_that("matches posterior for autocorrelated (non-independent) draws", {
  set.seed(42)
  niter <- 800
  nchains <- 4
  nvars <- 4
  raw <- array(rnorm(niter * nchains * nvars), dim = c(niter, nchains, nvars))
  # cumulative-sum random walk per chain/variable induces strong
  # autocorrelation, which is exactly what rhat/ess are meant to detect;
  # apply(..., cumsum) over the (nchains, nvars) margins preserves the
  # (niter, nchains, nvars) shape since cumsum returns a same-length vector
  ac <- apply(raw, c(2, 3), cumsum)
  x <- posterior::as_draws_array(ac)
  expect_matches_posterior(x)
})

test_that("matches posterior for draws with very low effective sample size", {
  # a slowly-mixing AR(1)-like chain: high autocorrelation -> low ESS, high rhat
  set.seed(7)
  niter <- 1000
  nchains <- 4
  nvars <- 2
  arr <- array(0, dim = c(niter, nchains, nvars))
  for (c in seq_len(nchains)) {
    for (v in seq_len(nvars)) {
      innov <- rnorm(niter)
      chain <- numeric(niter)
      chain[1] <- innov[1]
      for (t in 2:niter) chain[t] <- 0.98 * chain[t - 1] + innov[t]
      arr[, c, v] <- chain
    }
  }
  x <- posterior::as_draws_array(arr)
  expect_matches_posterior(x)
})

test_that("matches posterior when chains have different means (poor mixing / high rhat)", {
  set.seed(8)
  niter <- 500
  nchains <- 4
  nvars <- 3
  arr <- array(rnorm(niter * nchains * nvars), dim = c(niter, nchains, nvars))
  # shift each chain's mean so rhat should be large
  for (c in seq_len(nchains)) arr[, c, ] <- arr[, c, ] + c * 5
  x <- posterior::as_draws_array(arr)
  expect_matches_posterior(x)
})

test_that("matches posterior for larger nvars (exercises the TBB parallel path)", {
  x <- make_draws(300, 4, 250, seed = 9)
  expect_matches_posterior(x)
})

test_that("matches posterior with long chains", {
  x <- make_draws(20000, 4, 3, seed = 10)
  expect_matches_posterior(x)
})
