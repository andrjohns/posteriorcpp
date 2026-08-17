#' @keywords internal
.posteriorcpp_all_stats <- c(
  "mean", "median", "sd", "var", "mad", "q5", "q95",
  "rhat", "rhat_basic", "ess_bulk", "ess_tail", "ess_basic",
  "ess_mean", "ess_sd", "mcse_mean", "mcse_sd"
)

#' @keywords internal
.posteriorcpp_default_stats <- c(
  "mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail"
)

#' Fast summaries of posterior draws
#'
#' A C++/RcppEigen implementation of `posterior::summarise_draws()` computing
#' the default summary and convergence measures: mean, median, sd, mad, q5,
#' q95, rhat, ess_bulk, and ess_tail. Additional measures matching their
#' `posterior` equivalents are available opt-in via `stats` (see below).
#'
#' @param x A `draws` object, or an object coercible to one via
#'   [posterior::as_draws()].
#' @param stats Character vector of statistics to compute. The default is
#'   `"mean"`, `"median"`, `"sd"`, `"mad"`, `"q5"`, `"q95"`, `"rhat"`,
#'   `"ess_bulk"`, `"ess_tail"` — matching `posterior::summarise_draws()`'s
#'   own default set exactly. More are available opt-in, each matching its
#'   `posterior` equivalent: `"var"` (variance, i.e. `sd^2`); the unnormalised
#'   diagnostics `"rhat_basic"`/`"ess_basic"` (split-chains without the
#'   rank-normalising z-score step that `"rhat"`/`"ess_bulk"` apply);
#'   `"ess_mean"` (identical to `"ess_basic"`, the ESS of the mean
#'   estimator); `"ess_sd"` (ESS of the squared centered draws, i.e. of the
#'   SD estimator); and the Monte Carlo standard errors `"mcse_mean"`
#'   (`sd / sqrt(ess_mean)`) and `"mcse_sd"` (first-order Taylor
#'   approximation per Kenney & Keeping 1951). Output columns are always in
#'   the fixed canonical order above regardless of how `stats` is ordered.
#'   Statistics that share underlying work (e.g. `"rhat"` and `"ess_bulk"`
#'   both rank-normalise and split the chains; `"rhat_basic"` and `"rhat"`
#'   both start from the same unnormalised split; `"mcse_sd"` reuses
#'   `"ess_sd"`'s ESS as its own denominator) still only do that work once,
#'   but requesting fewer statistics skips work that only unrequested
#'   statistics need — e.g. `stats = c("mean", "sd")` skips sorting,
#'   rank-normalisation, and FFT autocovariance entirely.
#' @return A [tibble][tibble::tibble] with one row per variable and one
#'   column per requested statistic.
#' @export
summarise_draws <- function(x, stats = .posteriorcpp_default_stats) {
  if (length(stats) == 0L) {
    stop("`stats` must specify at least one statistic to compute.")
  }
  unknown <- setdiff(stats, .posteriorcpp_all_stats)
  if (length(unknown) > 0L) {
    stop(sprintf(
      "Unknown `stats`: %s. Must be one or more of: %s.",
      paste(unknown, collapse = ", "), paste(.posteriorcpp_all_stats, collapse = ", ")
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

  want <- .posteriorcpp_all_stats %in% stats
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
