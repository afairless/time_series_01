# data from:
#   https://otexts.com/fpp2/non-seasonal-arima.html
#   8.5 Non-seasonal ARIMA models
#   https://github.com/robjhyndman/fpp2-package/blob/master/data/uschange.rda


library(bayesforecast)


load_data <- function() {
  # Load 'Consumption' time series data from the 'uschange.rda' file

  input_filepath <- paste("..", "input", "uschange.rda", sep = "/")
  load(input_filepath)
  # head(uschange)
  # colnames(uschange)

  time_series <- as.vector(uschange[ , "Consumption"])

  return(time_series)

}


mkdir <- function(directory_path) {
  # Create directory at 'directory_path' if it does not exist

  if (!dir.exists(directory_path)) {
    dir.create(directory_path, recursive = TRUE)
  }

}


run_model <- function(
  time_series, order, seasonal_order, prior_ar, output_path, seed) {
  # Fit a bayesforecast/Stan SARIMA model with the given 'order' and 
  #   'seasonal_order' on the given 'time_series' data
  # Save the results in 'output_path'


  # RUN MODEL
  ##################################################

  set.seed(seed)
  model_result = stan_sarima(
    time_series, order=order, seasonal=seasonal_order,
    prior_ar=prior_ar,
    stepwise=F, chains=6, iter=16000, warmup=4000)


  # SAVE RESULTS
  ##################################################

  output_filepath = paste(output_path, "summary.csv", sep = "/")
  summary_df = summary(model_result)
  write.csv(summary_df, file=output_filepath, row.names=T)

  output_filepath = paste(output_path, "report.txt", sep = "/")
  sink(output_filepath)
  report(model_result)
  print(model_result)
  waic(model_result)
  sink()

}


main <- function() {


  output_path <- paste("..", "output", "s06_bayesforecast", sep = "/")

  time_series = load_data()


  # order, AR/p, d, MA/q
  order = c(1, 0, 3)
  # seasonal order, AR/P, D, MA/Q
  seasonal_order = c(0, 0, 0)


  # model with default AR prior
  ##################################################

  prior_ar = normal(0, 0.5)
  output_subpath <- paste(output_path, 'default_ar_prior', sep = "/")
  mkdir(output_subpath)
  run_model(
    time_series, order, seasonal_order, prior_ar, output_subpath, seed=874310)


  # model with fairly strong, "correct" AR prior
  ##################################################

  prior_ar = normal(0.6, 0.2)
  output_subpath <- paste(output_path, 'custom_ar_prior_01', sep = "/")
  mkdir(output_subpath)
  run_model(
    time_series, order, seasonal_order, prior_ar, output_subpath, seed=874310)

}


main()

