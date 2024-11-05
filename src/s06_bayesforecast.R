# data from:
#   https://otexts.com/fpp2/non-seasonal-arima.html
#   8.5 Non-seasonal ARIMA models
#   https://github.com/robjhyndman/fpp2-package/blob/master/data/uschange.rda


library(bayesforecast)


mkdir <- function(directory_path) {
  # Create directory at 'directory_path' if it does not exist

  if (!dir.exists(directory_path)) {
    dir.create(directory_path, recursive = TRUE)
  }

}

input_filepath <- paste("..", "input", "uschange.rda", sep = "/")
load(input_filepath)
# head(uschange)
# colnames(uschange)

# time_series = uschange[ , "Consumption"]
time_series <- as.vector(uschange[ , "Consumption"])

ar = 1
# order, AR/p, d, MA/q
order = c(ar, 0, 3)
# seasonal order, AR/P, D, MA/Q
seasonal_order = c(0, 0, 0)

output_path <- paste(
  "..", 
  "output", 
  "s06_bayesforecast", 
  paste("ar", as.character(ar), sep = ''), 
  sep = "/")

mkdir(output_path)

set.seed(874310)
model_result = stan_sarima(
  time_series, order=order, seasonal=seasonal_order,
  prior_ar=normal(0.6, 0.2), 
  stepwise=F, chains=6, iter=16000, warmup=4000)


class(model_result)
methods(class = class(model_result))





# to df
summary(model_result)

# to text
report(model_result)
# prior_summary(model_result)
# model(model_result)
print(model_result)
waic(model_result)

