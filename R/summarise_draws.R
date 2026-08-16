#' Fast summaries of posterior draws
#'
#' A C++/RcppEigen implementation of `posterior::summarise_draws()` computing
#' the default summary and convergence measures: mean, median, sd, mad, q5,
#' q95, rhat, ess_bulk, and ess_tail.
#'
#' @param x A `draws` object, or an object coercible to one via
#'   [posterior::as_draws()].
#' @return A [tibble][tibble::tibble] with one row per variable.
#' @export
summarise_draws <- function(x) {
  x <- posterior::as_draws(x)
  x <- posterior::repair_draws(x)
  x <- posterior::as_draws_array(x)

  vars <- posterior::variables(x)
  nvars <- length(vars)
  if (posterior::ndraws(x) == 0L || nvars == 0L) {
    return(structure(
      list(variable = character()),
      row.names = c(NA_integer_, 0L),
      class = c("tbl_df", "tbl", "data.frame")
    ))
  }

  out <- summarise_draws_cpp(unclass(x))
  out$variable <- vars

  # Fast S3 tibble construction without overhead
  structure(
    out[c("variable", "mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail")],
    row.names = c(NA_integer_, -nvars),
    class = c("tbl_df", "tbl", "data.frame")
  )
}

#' @rdname summarise_draws
#' @export
summarize_draws <- summarise_draws
