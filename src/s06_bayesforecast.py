#! /usr/bin/env python3

# https://github.com/asael697/bayesforecast/blob/master/inst/stan/Sarima.stan 
# https://github.com/asael697/bayesforecast/blob/master/R/Sarima.R

import numpy as np
import polars as pl
import pyreadr
from pathlib import Path

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
    Load 'Consumption' time series data from the 'uschange.rda' file
    """

    input_filepath = input_path / 'uschange.rda'
    df = pyreadr.read_r(input_filepath)['uschange']
    df = pl.DataFrame(pyreadr.read_r(input_filepath)['uschange'])
    time_series = df['Consumption'].to_numpy()

    return time_series


def run_sarimax_model(
    time_series: np.ndarray, order: tuple[int, int, int]
    ) -> sarimax.SARIMAXResultsWrapper:
    """
    Run SARIMAX model on the downloaded time series data from 'fpp2' textbook
    """

    model_result = sarimax.SARIMAX(time_series, order=order, trend='c').fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    return model_result


def main():
    """
    Save results of SARIMAX model for ARMA order used in 'fpp2' textbook 
    """


    # SET PATHS AND DATA
    ##################################################

    input_path = Path.cwd() / 'input'
    output_path = Path.cwd() / 'output' / 's06_bayesforecast' / 'statsmodels'
    output_path.mkdir(exist_ok=True, parents=True)

    time_series = load_data(input_path)


    # RUN MODEL
    ##################################################

    # order, AR/p, d, MA/q
    order = (1, 0, 3)
    # seasonal order, AR/P, D, MA/Q
    seasonal_order = (0, 0, 0)

    sarimax_result = run_sarimax_model(time_series, order)


    # SAVE MODEL RESULTS
    ##################################################

    output_filepath = output_path / 'sarimax_summary.md'
    md = []
    md.append('## SARIMAX model results')
    md.append(sarimax_result.summary().as_html())
    write_list_to_text_file(md, output_filepath, True)

    output_filepath = output_path / 'sarimax_parameters.csv'
    result_df = pl.DataFrame(
        {'param_names': sarimax_result.param_names, 
         'params': sarimax_result.params})
    result_df.write_csv(output_filepath)


if __name__ == '__main__':
    main()
