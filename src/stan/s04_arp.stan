// from Stan User's Guide
// https://mc-stan.org/docs/stan-users-guide/index.html
// Stan User’s Guide
// Version 2.35

// https://mc-stan.org/docs/stan-users-guide/time-series.html
// Time-Series Models

// NOTE:  running model with beta as vector or array gets similar results, but
//  the array declaration allows range constraints with 'lower' and 'upper'

data {
  int<lower=0> P;        // autoregressive order
  int<lower=0> T;        // number of time steps in series
  array[T] real y;       // time series values
}
parameters {
  real alpha;
  //vector[P] beta;
  array[P] real<lower=-1, upper=1> beta;
  real<lower=0> sigma;
}
model {
  for (t in (P+1):T) {
    real mu = alpha;
    for (p in 1:P) {
      mu += beta[p] * y[t-p];
    }
    y[t] ~ normal(mu, sigma);
  }
}
