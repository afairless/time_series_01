// from Stan User's Guide
// https://mc-stan.org/docs/stan-users-guide/index.html
// Stan User’s Guide
// Version 2.35

// https://mc-stan.org/docs/stan-users-guide/time-series.html
// Time-Series Models

data {
  int<lower=2> T;         // number of time steps in series
  vector[T] y;            // time series values
}
parameters {
  real mu;                // mean
  real<lower=0> sigma;    // error scale
  real theta;             // error coefficient
}
transformed parameters {

  vector[T] epsilon;      // error terms

  epsilon[1] = y[1] - mu;

  for (t in 2:T) {
    epsilon[t] = y[t] - mu - theta * epsilon[t-1];
  }
}
model {

  mu ~ cauchy(0, 2.5);
  theta ~ cauchy(0, 2.5);
  sigma ~ normal(0, 2.5);

  for (t in 2:T) {
    y[t] ~ normal(mu + theta * epsilon[t-1], sigma);
  }
}
