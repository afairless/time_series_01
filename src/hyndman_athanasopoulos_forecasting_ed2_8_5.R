# ~/R-4.4.1/bin/R

input_filepath <- paste("..", "input", "uschange.rda", sep = "/")
load(input_filepath)

uschange

head(uschange)

colnames(uschange)

uschange[, "Consumption"]

consumption <- as.vector(uschange[, "Consumption"])
