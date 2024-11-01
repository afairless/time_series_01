#! /usr/bin/env python3

import numpy as np
import polars as pl
import pyreadr
from pathlib import Path

from cmdstanpy import CmdStanModel
from cmdstanpy.stanfit.mcmc import CmdStanMCMC
import statsmodels.tsa.statespace.sarimax as sarimax


try:
    from src.common import (
        write_list_to_text_file,
        )

except:
    from common import (
        write_list_to_text_file,
        )


def load_data(input_path: Path) -> np.ndarray:
    """
    Load data from 'uschange.rda' file
    """

    input_filepath = input_path / 'uschange.rda'
    df = pyreadr.read_r(input_filepath)['uschange']
    df = pl.DataFrame(pyreadr.read_r(input_filepath)['uschange'])
    time_series = df['Consumption'].to_numpy()

    return time_series


def run_sarimax_model(time_series: np.ndarray) -> sarimax.SARIMAXResultsWrapper:
    """
    Run SARIMAX model on the downloaded time series data from 'fpp2' textbook
    """

    # order, AR/p, d, MA/q
    order = (0, 0, 1)
    model_result = sarimax.SARIMAX(time_series, order=order, trend='c').fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    return model_result


def run_stan_model(
    stan_path: Path, time_series: np.ndarray, 
    output_path: Path) -> CmdStanMCMC:

    stan_data = {
        'T': len(time_series), 
        'y': time_series}

    stan_filename = 's02_ma1.stan'
    stan_filepath = stan_path / stan_filename
    model = CmdStanModel(stan_file=stan_filepath)

    fit_model = model.sample(
        data=stan_data, chains=4, thin=2, seed=21520,
        iter_warmup=4000, iter_sampling=12000, output_dir=output_path)


    # tabular summaries
    ##################################################

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_df = fit_model.summary()
    output_filepath = output_path / 'summary.csv'
    fit_df.to_csv(output_filepath, index=True)


    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter_sampling' in 'CmdStanModel.sample' call
    draws_df = fit_model.draws_pd()
    output_filepath = output_path / 'draws.csv'
    draws_df.to_csv(output_filepath, index=True)

    return fit_model


def compare_sarimax_stan_models(sarimax_result, stan_result):
    """
    Compare coefficients for SARIMAX and Stan models
    """

    fit_df = stan_result.summary()

    # mu/intercept
    assert np.isclose(fit_df.iloc[1, 0], sarimax_result.params[0], atol=1e-2)

    # theta/ma.L1
    assert np.isclose(fit_df.iloc[3, 0], sarimax_result.params[1], atol=1e-2)

    # sigma
    assert np.isclose(
        fit_df.iloc[2, 0], 
        np.sqrt(sarimax_result.params[2]), atol=1e-2)

    print('Results of SARIMAX and Stan models approximately match')


def report_model_result_comparison(
    output_path: Path, model_result: sarimax.SARIMAXResultsWrapper):
    """
    Save time series plots and model results from 'fpp2' textbook and SARIMAX
        model in markdown file
    """

    md_filepath = output_path / 'report.md'
    md = []

    md.append('## SARIMAX model results')

    md.append(model_result.summary().as_html())

    write_list_to_text_file(md, md_filepath, True)


def main():
    """
    Compare Stan model that accommodates MA(1) with corresponding SARIMAX model
        to ensure that Stan model is coded correctly
    """

    # set paths
    ##################################################

    input_path = Path.cwd() / 'input'

    src_path = Path.cwd() / 'src'
    stan_path = src_path / 'stan'

    output_path = Path.cwd() / 'output' / 's02_ma1'
    output_path.mkdir(exist_ok=True, parents=True)


    # run and compare models
    ##################################################

    time_series = load_data(input_path)
    sarimax_result = run_sarimax_model(time_series)
    stan_result = run_stan_model(stan_path, time_series, output_path)
    compare_sarimax_stan_models(sarimax_result, stan_result)

    report_model_result_comparison(output_path, sarimax_result)


if __name__ == '__main__':
    main()
