#' @keywords internal
.posteriorcpp_all_stats <- c(
  "mean",
  "median",
  "sd",
  "var",
  "mad",
  "q5",
  "q95",
  "rhat",
  "rhat_basic",
  "ess_bulk",
  "ess_tail",
  "ess_basic",
  "ess_mean",
  "ess_sd",
  "mcse_mean",
  "mcse_sd"
)

#' @keywords internal
.posteriorcpp_default_stats <- c(
  "mean",
  "median",
  "sd",
  "mad",
  "q5",
  "q95",
  "rhat",
  "ess_bulk",
  "ess_tail"
)

#' Fast summaries of posterior draws
#'
#' A C++/Eigen implementation of `posterior::summarise_draws()`.
#'
#' @param x A `draws` object, or an object coercible to one via
#'   [posterior::as_draws()].
#' @param stats Character vector of statistics to compute. Defaults to
#'   `"mean"`, `"median"`, `"sd"`, `"mad"`, `"q5"`, `"q95"`, `"rhat"`,
#'   `"ess_bulk"`, `"ess_tail"`. Additional opt-in stats matching their
#'   `posterior` equivalents: `"var"`, `"rhat_basic"`, `"ess_basic"`,
#'   `"ess_mean"`, `"ess_sd"`, `"mcse_mean"`, `"mcse_sd"`. Output columns
#'   always follow the fixed canonical order regardless of `stats` order.
#' @return A [tibble][tibble::tibble] with one row per variable and one
#'   column per requested statistic.
#' @export
summarise_draws_cpp <- function(x, stats = .posteriorcpp_default_stats) {
  if (length(stats) == 0L) {
    stop("`stats` must specify at least one statistic to compute.")
  }
  unknown <- setdiff(stats, .posteriorcpp_all_stats)
  if (length(unknown) > 0L) {
    stop(sprintf(
      "Unknown `stats`: %s. Must be one or more of: %s.",
      paste(unknown, collapse = ", "),
      paste(.posteriorcpp_all_stats, collapse = ", ")
    ))
  }
  stats <- .posteriorcpp_all_stats[.posteriorcpp_all_stats %in% stats]

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

  out <- summarise_draws_cpp_(unclass(x), stats)
  out$variable <- vars

  structure(
    out[c("variable", stats)],
    row.names = c(NA_integer_, -nvars),
    class = c("tbl_df", "tbl", "data.frame")
  )
}
