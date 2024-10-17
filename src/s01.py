#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
import pyreadr
from pathlib import Path

from cmdstanpy import CmdStanModel
import statsmodels.tsa.statespace.sarimax as sarimax

import arviz as az
import matplotlib.pyplot as plt

try:
    from src.common import (
        plot_time_series,
        plot_time_series_autocorrelation,
        )

except:
    from common import (
        plot_time_series,
        plot_time_series_autocorrelation,
        )


def transcribe_fpp2_8_5_model_results() -> tuple[pl.DataFrame, dict[str, float]]:
    """
    Model results from:

        https://otexts.com/fpp2/non-seasonal-arima.html
        8.5 Non-seasonal ARIMA models
    """

    coef_dict = {
        'label': ['ar1', 'ma1', 'ma2', 'ma3', 'mean'],
        'coef':  [0.589, -0.353, 0.085, 0.174, 0.745],
        'se':    [0.154,  0.166, 0.082, 0.084, 0.093]}
    coef_df = pl.DataFrame(coef_dict)

    summary_dict = {
        'sigma^2': 0.35,
        'log_likelihood': -164.8,
        'AIC': 341.6,
        'AICc': 342.1,
        'BIC': 361,
        'c': 0.307}

    return coef_df, summary_dict


def main():

    input_path = Path.cwd() / 'input'
    input_filepath = input_path / 'uschange.rda'
    df = pyreadr.read_r(input_filepath)['uschange']
    df = pl.DataFrame(pyreadr.read_r(input_filepath)['uschange'])

    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing.md'
    md = []

    time_series = df['Consumption'].to_numpy()

    # order, AR/p, d, MA/q
    order = (1, 0, 3)
    model_result = sarimax.SARIMAX(time_series, order=order, trend='c').fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    model_result.summary()
    dir(model_result)

    coef_df, summary_dict = transcribe_fpp2_8_5_model_results()

    # the ARMA coefficients from the fpp2 and SARIMAX models match closely
    arma_coef_len = 4
    arma_coef_sarimax = model_result.params[1:1+arma_coef_len]
    arma_coef_fpp2 = coef_df['coef'][:arma_coef_len]
    assert ((arma_coef_sarimax - arma_coef_fpp2) < 1e-3).all()

    # the constants from the fpp2 and SARIMAX models match
    constant_sarimax = model_result.params[0]
    constant_coef_fpp2 = coef_df['coef'][arma_coef_len]
    constant_fpp2 = constant_coef_fpp2 * (1 - arma_coef_fpp2[0])
    assert np.isclose(constant_sarimax, constant_fpp2, atol=1e-3)

    # summary statistics from the fpp2 and SARIMAX models match
    assert np.isclose(summary_dict['AIC'], model_result.aic, atol=1e-1)
    assert np.isclose(summary_dict['AICc'], model_result.aicc, atol=1e-1)
    assert np.isclose(summary_dict['BIC'], model_result.bic, atol=1e-1)
    assert np.isclose(
        summary_dict['log_likelihood'], model_result.llf, atol=1e-1)
    assert np.isclose(
        summary_dict['sigma^2'], model_result.params[-1], atol=1e-1)


    plt.plot(time_series)
    plt.show()
    plt.clf()
    plt.close()









if __name__ == '__main__':
    main()
