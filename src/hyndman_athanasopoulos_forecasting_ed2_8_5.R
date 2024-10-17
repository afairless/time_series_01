# data from:
#   https://otexts.com/fpp2/non-seasonal-arima.html
#   8.5 Non-seasonal ARIMA models
#   https://github.com/robjhyndman/fpp2-package/blob/master/data/uschange.rda

# ~/R-4.4.1/bin/R

input_filepath <- paste("..", "input", "uschange.rda", sep = "/")
load(input_filepath)

uschange

head(uschange)

colnames(uschange)

uschange[, "Consumption"]

consumption <- as.vector(uschange[, "Consumption"])
