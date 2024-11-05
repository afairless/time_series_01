#! /usr/bin/env python3

import numpy as np
import polars as pl
import pyreadr
from pathlib import Path

import statsmodels.tsa.statespace.sarimax as sarimax

try:
    from src.common import (
        write_list_to_text_file,
        plot_time_series,
        transcribe_fpp2_8_5_model_results,
        )

except:
    from common import (
        write_list_to_text_file,
        plot_time_series,
        transcribe_fpp2_8_5_model_results,
        )


def run_sarimax_model(
    input_path: Path) -> tuple[np.ndarray, sarimax.SARIMAXResultsWrapper]:
    """
    Run SARIMAX model on the downloaded time series data from 'fpp2' textbook
    """

    input_filepath = input_path / 'uschange.rda'
    df = pyreadr.read_r(input_filepath)['uschange']
    df = pl.DataFrame(pyreadr.read_r(input_filepath)['uschange'])

    time_series = df['Consumption'].to_numpy()

    # order, AR/p, d, MA/q
    order = (1, 0, 3)
    model_result = sarimax.SARIMAX(time_series, order=order, trend='c').fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    return time_series, model_result


def compare_model_results(model_result: sarimax.SARIMAXResultsWrapper):
    """
    Verify that results from sarimax model match results from the 'fpp2' 
        textbook example
    """

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


def report_model_result_comparison(
    input_path: Path, output_path: Path, time_series_plot_filepath: Path,
    model_result: sarimax.SARIMAXResultsWrapper):
    """
    Save time series plots and model results from 'fpp2' textbook and SARIMAX
        model in markdown file
    """

    md_filepath = output_path / 'report.md'
    md = []

    md.append(
        '## Downloaded time series data looks identical to Figure 8.7 in '
        '"fpp2" textbook')
    figure_8_7_filepath = input_path / 'fpp2_book' / 'Figure_8.7.png'
    figure_8_7_relative_filepath = (
        figure_8_7_filepath.__str__().replace(Path.cwd().__str__(), '../..'))
    md.append('\n')
    md.append('![Image](' + figure_8_7_relative_filepath + '){width=640}')
    md.append('\n')
    md.append('![Image](' + time_series_plot_filepath.name + '){width=640}')
    md.append('\n')


    md.append('## Results of "fpp2" textbook and SARIMAX models match')
    fpp2_results_filepath = (
        input_path / 'fpp2_book' / 'ARIMA(1,0,3)_results.png')
    fpp2_results_relative_filepath = (
        fpp2_results_filepath.__str__().replace(Path.cwd().__str__(), '../..'))
    md.append('\n')
    md.append('![Image](' + fpp2_results_relative_filepath + '){width=640}')
    md.append('\n')

    md.append(model_result.summary().as_html())

    write_list_to_text_file(md, md_filepath, True)


def main():

    input_path = Path.cwd() / 'input'

    output_path = Path.cwd() / 'output' / 's01'
    output_path.mkdir(exist_ok=True, parents=True)

    time_series, model_result = run_sarimax_model(input_path)
    compare_model_results(model_result)

    output_filepath = output_path / 'time_series.png'
    plot_time_series(
        time_series.reshape(1, -1),
        title='Downloaded time series',
        output_filepath=output_filepath)

    report_model_result_comparison(
        input_path, output_path, output_filepath, model_result)


if __name__ == '__main__':
    main()
