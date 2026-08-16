#' @keywords internal
.bayescpp_all_stats <- c("mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail")

#' Fast summaries of posterior draws
#'
#' A C++/RcppEigen implementation of `posterior::summarise_draws()` computing
#' the default summary and convergence measures: mean, median, sd, mad, q5,
#' q95, rhat, ess_bulk, and ess_tail.
#'
#' @param x A `draws` object, or an object coercible to one via
#'   [posterior::as_draws()].
#' @param stats Character vector of statistics to compute, a subset of
#'   `"mean"`, `"median"`, `"sd"`, `"mad"`, `"q5"`, `"q95"`, `"rhat"`,
#'   `"ess_bulk"`, `"ess_tail"` (the default: all of them, in that order
#'   regardless of how `stats` is ordered). Statistics that share underlying
#'   work (e.g. `"rhat"` and `"ess_bulk"` both rank-normalise and split the
#'   chains) still only do that work once, but requesting fewer statistics
#'   skips work that only unrequested statistics need — e.g.
#'   `stats = c("mean", "sd")` skips sorting, rank-normalisation, and FFT
#'   autocovariance entirely.
#' @return A [tibble][tibble::tibble] with one row per variable and one
#'   column per requested statistic.
#' @export
summarise_draws <- function(x, stats = .bayescpp_all_stats) {
  if (length(stats) == 0L) {
    stop("`stats` must specify at least one statistic to compute.")
  }
  unknown <- setdiff(stats, .bayescpp_all_stats)
  if (length(unknown) > 0L) {
    stop(sprintf(
      "Unknown `stats`: %s. Must be one or more of: %s.",
      paste(unknown, collapse = ", "), paste(.bayescpp_all_stats, collapse = ", ")
    ))
  }
  stats <- .bayescpp_all_stats[.bayescpp_all_stats %in% stats]

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

  want <- .bayescpp_all_stats %in% stats
  out <- summarise_draws_cpp(unclass(x), want)
  out$variable <- vars

  # Fast S3 tibble construction without overhead
  structure(
    out[c("variable", stats)],
    row.names = c(NA_integer_, -nvars),
    class = c("tbl_df", "tbl", "data.frame")
  )
}

#' @rdname summarise_draws
#' @export
summarize_draws <- summarise_draws
