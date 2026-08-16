test_that("returns a tibble with one row per variable and the default columns", {
  x <- make_draws(200, 4, 5, seed = 1)
  out <- bayescpp::summarise_draws(x)

  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 5)
  expect_identical(names(out), c("variable", all_stats))
  expect_identical(out$variable, posterior::variables(x))
})

test_that("column types are as expected", {
  x <- make_draws(200, 4, 3, seed = 2)
  out <- bayescpp::summarise_draws(x)

  expect_type(out$variable, "character")
  for (cn in all_stats) {
    expect_type(out[[cn]], "double")
  }
})

test_that("summarize_draws (US spelling) is identical to summarise_draws", {
  x <- make_draws(200, 4, 4, seed = 3)
  expect_identical(bayescpp::summarise_draws(x), bayescpp::summarize_draws(x))

  x2 <- make_draws(200, 4, 4, seed = 4)
  expect_identical(
    bayescpp::summarise_draws(x2, stats = c("mean", "rhat")),
    bayescpp::summarize_draws(x2, stats = c("mean", "rhat"))
  )
})

test_that("accepts draws-coercible input types, not just draws_array", {
  set.seed(5)
  m <- matrix(rnorm(400 * 3), nrow = 400, ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
  dm <- posterior::as_draws_matrix(m)

  expect_matches_posterior(dm)

  df <- posterior::as_draws_df(dm)
  expect_matches_posterior(df)
})

test_that("empty draws object returns an empty tibble with just the variable column", {
  x <- make_draws(100, 2, 0, seed = 6)
  out <- bayescpp::summarise_draws(x)
  expect_equal(nrow(out), 0)
  expect_identical(names(out), "variable")
})
