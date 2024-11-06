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

  time_series <- as.vector(uschange[, "Consumption"])

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
  model_result <- stan_sarima(
    time_series,
    order = order, seasonal = seasonal_order,
    prior_ar = prior_ar,
    stepwise = FALSE, chains = 6, iter = 16000, warmup = 4000)


  # SAVE RESULTS
  ##################################################

  output_filepath <- paste(output_path, "summary.csv", sep = "/")
  summary_df <- summary(model_result)
  write.csv(summary_df, file = output_filepath, row.names = TRUE)

  output_filepath <- paste(output_path, "report.txt", sep = "/")
  sink(output_filepath)
  report(model_result)
  print(model_result)
  waic(model_result)
  sink()

  return(summary_df)
}


main <- function() {
  # Fit Stan models with 'bayesforecast' package to time series data from 'fpp2' 
  #   textbook
  # Compare Stan models' results to non-Bayesian results from 'statsmodels' 
  #   SARIMAX model with a plot of each model's coefficients
  # Results:  the default priors for 'bayesforecast' do not match the SARIMAX
  #   results well, but a strong prior on the AR term provides a similar fit 

  output_path <- paste("..", "output", "s06_bayesforecast", sep = "/")

  time_series <- load_data()


  # order, AR/p, d, MA/q
  order <- c(1, 0, 3)
  # seasonal order, AR/P, D, MA/Q
  seasonal_order <- c(0, 0, 0)


  # model with default AR prior
  ##################################################

  prior_ar <- normal(0, 0.5)
  output_subpath <- paste(output_path, "default_ar_prior", sep = "/")
  mkdir(output_subpath)
  default_prior_summary_df <- run_model(
    time_series, order, seasonal_order, prior_ar, output_subpath,
    seed = 874310)


  # model with fairly strong, "correct" AR prior
  ##################################################

  prior_ar <- normal(0.6, 0.2)
  output_subpath <- paste(output_path, "custom_ar_prior_01", sep = "/")
  mkdir(output_subpath)
  strong_prior_summary_df <- run_model(
    time_series, order, seasonal_order, prior_ar, output_subpath,
    seed = 874310)


  # compare model results above with 'statsmodels' SARIMAX model
  ##################################################

  input_filepath <- paste(
    output_path, "statsmodels", "sarimax_parameters.csv",
    sep = "/")
  statsmodels_df <- read.csv(input_filepath)

  # re-arrange 'statsmodels' result to match other models' results format
  statsmodels_df2 <- rbind(
    statsmodels_df[1, ],
    statsmodels_df[dim(statsmodels_df)[1], ],
    statsmodels_df[2:(dim(statsmodels_df)[1] - 1), ])

  # switch variance to standard deviation
  statsmodels_df2[2, "params"] <- sqrt(statsmodels_df2[2, "params"])

  # compile results from all models into a single dataframe
  param_n <- dim(statsmodels_df)[1]
  all_param_df <- cbind(
    statsmodels_df2[, "params"],
    default_prior_summary_df[1:param_n, "mean"],
    strong_prior_summary_df[1:param_n, "mean"])

  # add row and column names
  rownames(all_param_df) <- rownames(default_prior_summary_df)[1:param_n]
  colnames(all_param_df) <- c(
    "statsmodels", "bayes_default_prior", "bayes_strong_prior")

  # plot model coefficients
  plot_filename = "params_by_model.png"
  output_filepath <- paste(output_path, plot_filename, sep = "/")
  png(output_filepath, width = 640, height = 480)
  barplot(
    t(all_param_df),
    beside = TRUE,
    xlab = rownames(all_param_df),
    legend = colnames(all_param_df))
  dev.off()

  output_filepath <- paste(output_path, "report.md", sep = "/")
  sink(output_filepath)
    cat(
      paste(
        "Results:  the default priors for 'bayesforecast' do not match the ",
        "SARIMAX results well, but a strong prior on the AR term provides a ",
        "similar fit\n"))
    cat(paste('![Image](', plot_filename, '){width=640}'))
  sink()

}


main()
