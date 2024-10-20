// from Stan User's Guide
// https://mc-stan.org/docs/stan-users-guide/index.html
// Stan User’s Guide
// Version 2.35

// https://mc-stan.org/docs/stan-users-guide/time-series.html
// Time-Series Models

data {
  int<lower=0> T;        // number of time steps in series
  vector[T] y;           // time series values
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y[2:T] ~ normal(alpha + beta * y[1:(T-1)], sigma);
}
