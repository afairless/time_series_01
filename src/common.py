#! /usr/bin/env python3

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, fields

import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace.sarimax as sarimax
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import (
    root_mean_squared_error as skl_rmse, 
    mean_absolute_error as skl_mae, 
    median_absolute_error as skl_mdae)


@dataclass
class TimeSeriesDifferencing:

    k_diff: int = 1
    k_seasonal_diff: int = 0
    seasonal_periods: int = 1
    original_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))
    final_difference_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))
    prepend_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))


    def difference_time_series_seasonal(
        self, series: np.ndarray, k_seasonal_diff: int, seasonal_periods: int
        ) -> np.ndarray:
        """
        Apply seasonal differencing to given time series vector
        """

        for _ in range(k_seasonal_diff):
            self.prepend_vector = np.append(
                self.prepend_vector, series[:seasonal_periods])
            series = (
                series[seasonal_periods:] - 
                series[:-seasonal_periods])

        return series


    def difference_time_series(self, series: np.ndarray) -> np.ndarray:
        """
        Apply simple and/or seasonal differencing to given time series vector
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert len(series) >= max(self.k_diff, self.k_seasonal_diff) 
        if self.k_seasonal_diff > 0:
            assert self.seasonal_periods >= 1

        series = series.copy()
        series = series.reshape(-1)
        self.original_vector = series

        # seasonal differencing
        series = self.difference_time_series_seasonal(
            series, self.k_seasonal_diff, self.seasonal_periods)

        # simple/ordinary differencing
        diff_vectors = [
            np.diff(series, k, axis=0) for k in range(self.k_diff+1)]
        series = diff_vectors[-1] 
        prepends = np.array([dfv[0] for dfv in diff_vectors[:-1]])
        self.prepend_vector = np.append(self.prepend_vector, prepends)
        self.final_difference_vector = series

        return series


    def _difference_time_series_2(self, series: np.ndarray) -> np.ndarray:
        """
        Apply simple and/or seasonal differencing to given time series vector
        This function reverses the order of differencing:  simple, then 
            seasonal, instead of seasonal, then simple
        This reversal is used in tests to verify that either order of 
            differencing produces the same result
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert (self.k_diff + self.k_seasonal_diff) <= len(series)
        if self.k_seasonal_diff > 0:
            assert self.seasonal_periods >= 1

        series = series.copy()
        series = series.reshape(-1)
        self.original_vector = series

        # simple/ordinary differencing
        series = np.diff(series, self.k_diff, axis=0)
        self.final_difference_vector = series

        # seasonal differencing
        series = self.difference_time_series_seasonal(
            series, self.k_seasonal_diff, self.seasonal_periods)

        return series


    def periodic_cumulative_sum(
        self, series: np.ndarray, seasonal_periods: int) -> np.ndarray:
        """
        Calculate cumulative sums in a 1-dimensional array 'series' by position 
            in a period specified by 'seasonal_periods'
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert len(series) > seasonal_periods

        periods_n = len(series) - seasonal_periods
        periodic_cumsum = series[:seasonal_periods]
        for idx_1 in range(periods_n):
            idx_2 = idx_1 + seasonal_periods
            period_sum = np.array([periodic_cumsum[idx_1] + series[idx_2]])
            periodic_cumsum = np.concatenate([periodic_cumsum, period_sum])

        return periodic_cumsum  


    def de_difference_time_series(
        self, series: np.ndarray=np.array([])) -> np.ndarray:
        """
        "De-difference" given time series vector, i.e., back-transform the 
            series so that the "forward" differencing procedure is reversed
        'De-differencing' is often referred to as 'integrating', though Box, 
            Jenkins, and Reinsel (Time Series Analysis:  Forecasting and 
            Control, 3rd edition, Prentice Hall, Inc., 1994) on page 12 suggest
            that the better term is 'summing'

        NOTE:  as cumulative sums are added from the start to the end of the
            vector, error from floating-point imprecision accumulates
        """

        # INPUT PRE-CHECKS
        ##################################################

        if self.original_vector.size == 0:
            raise ValueError(
                'Original time series vector has not been provided; '
                'run method "difference_time_series" on the vector first')

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1
        # if (np.array(series.shape) > 1).sum() > 1:
        #     raise ValueError('Input series must be a 1-dimensional array')

        if series.size == 0:
            return self.original_vector

        if self.k_diff == 0 and self.k_seasonal_diff == 0:
            return series

        series = series.copy()
        series = series.reshape(-1)
        original_vector = self.original_vector.copy()
        original_vector = original_vector.reshape(-1)

        season_period_diff_len = self.k_seasonal_diff * self.seasonal_periods
        diff_total = self.k_diff + season_period_diff_len 
        assert (len(series) + diff_total) == len(original_vector)


        # DE-DIFFERENCE TIME SERIES
        ##################################################

        # if the given series is the final difference vector, pass original
        #   difference vector along as the combined vector
        if np.allclose(self.final_difference_vector, series):
            # could return 'original_vector' here for speed, but continuing
            #   through rest of code provides an important debugging scenario
            combined_vector = series.copy()
        # otherwise, sum the given vector with the final difference vector, 
        #   i.e., the given vector modifies the original differences
        else:
            combined_vector = np.sum(
                [series, self.final_difference_vector], axis=0)

        # simple de-differencing
        p = None
        for p in range(-1, -self.k_diff-1, -1):
            prepend = np.array([self.prepend_vector[p]])
            prepend_vector = np.concatenate([prepend, combined_vector])
            combined_vector = np.cumsum(prepend_vector)

        # remove/"pop" used elements from 'prepend_vector'
        self.prepend_vector = self.prepend_vector[:p] 

        for _ in range(self.k_seasonal_diff):

            # end_idx = self.k_seasonal_diff * self.seasonal_periods
            end_idx = len(self.prepend_vector)
            start_idx = end_idx - self.seasonal_periods
            prepend = self.prepend_vector[start_idx:end_idx]
            prepend_vector = np.concatenate([prepend, combined_vector])
            combined_vector = self.periodic_cumulative_sum(
                prepend_vector, self.seasonal_periods)

            # remove/"pop" used elements from 'prepend_vector'
            self.prepend_vector = self.prepend_vector[:-self.seasonal_periods]

        return combined_vector  


def get_git_root_path() -> Path | None:
    """
    Returns the top-level project directory where the Git repository is defined
    """

    try:
        # Run the git command to get the top-level directory
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], 
            stderr=subprocess.STDOUT)
        git_root_path = Path(git_root.decode('utf-8').strip())
        return git_root_path 

    except subprocess.CalledProcessError as e:
        print('Error while trying to find the Git root:', e.output.decode())
        return None


def write_list_to_text_file(
    a_list: list[Path | str], text_filename: Path | str, overwrite: bool=False):
    """
    Writes a list of strings to a text file
    If 'overwrite' is 'True', any existing file by the name of 'text_filename'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file by the name of 'text_filename'

    :param a_list: a list of strings to be written to a text file
    :param text_filename: a string denoting the filepath or filename of text
        file
    :param overwrite: Boolean indicating whether to overwrite any existing text
        file or to append 'a_list' to that file's contents
    :return:
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    with open(text_filename, append_or_overwrite, encoding='utf-8') as txt_file:
        for e in a_list:
            txt_file.write(str(e))
            txt_file.write('\n')


def convert_path_to_relative_path_str(path: Path) -> str:
    return str(path).replace(str(Path.cwd()), '.')


##################################################
# TIME SERIES METRICS
##################################################

def root_median_squared_error(y_true, y_pred):
    return np.sqrt(np.median((y_true - y_pred) ** 2))


@dataclass
class TimeSeriesMetrics:
    rmse: float
    rmdse: float
    mae: float
    mdae: float


def calculate_time_series_metrics(
    series_1: np.ndarray, series_2: np.ndarray) -> TimeSeriesMetrics:
    """
    Calculate basic error metrics for two time series
    """
    
    # if one series is shorter than the other, remove elements from the start
    #   of the longer series so that the two series are the same length, then
    #   calculate metrics
    min_len = min(len(series_1), len(series_2))
    series_1 = series_1[-min_len:]
    series_2 = series_2[-min_len:]

    # calculate metrics
    rmse = skl_rmse(series_1, series_2).item()
    rmdse = root_median_squared_error(series_1, series_2).item()
    mae = skl_mae(series_1, series_2).item()
    mdae = skl_mdae(series_1, series_2).item()

    metrics = TimeSeriesMetrics(rmse, rmdse, mae, mdae)

    return metrics


def is_array_one_dimensional(arr: np.ndarray) -> bool:
    """
    Check if the array is one-dimensional
    """
    return (np.array(arr.shape) > 1).sum() <= 1


def decompose_and_forecast_seasonal_naive(
    time_series: np.ndarray, test_start_idx: int, last_observation: float,
    period: int, decompose_additive: bool, plot_decomposition: bool=False,
    output_filepath: Path=Path('plot.png')) -> np.ndarray:
    """
    Produce seasonal naive forecast by decomposing the time series into trend
        and seasonal components

    time_series - a 1-dimensional array representing the time series
    test_start_idx - the index of the time series at which the test set starts
    last_fit_observation - the last observation before the start of the forecast
        time range; it serves as a constant level for the seasonal naive 
        forecast
    period - the period of the seasonal component
    decompose_additive - whether to decompose the time series using an additive 
        model ('True') or a multiplicative model ('False')
    plot_decomposition - whether to save a plot of the decomposition
    output_filepath - the filepath at which to save the decomposition plot
    """

    # INPUT PRE-CHECKS AND SETTINGS
    ##################################################

    assert is_array_one_dimensional(time_series)
    assert test_start_idx > 0
    assert test_start_idx < len(time_series)

    assert period < len(time_series) // 2

    if decompose_additive:
        decompose_model = 'additive'
    else:
        decompose_model = 'multiplicative'


    # DECOMPOSE TIME SERIES AND GENERATE FORECAST
    ##################################################

    decomposition = seasonal_decompose(
        time_series, model=decompose_model, period=period, two_sided=False)

    time_series[test_start_idx-5:test_start_idx+5]
    test_forecast_seasonal_naive = np.add(
        last_observation,
        decomposition.seasonal[test_start_idx:])

    # replace NaNs with last observation carried forward
    nan_mask = np.isnan(test_forecast_seasonal_naive)
    for idx in np.where(nan_mask)[0]:
        test_forecast_seasonal_naive[idx] = test_forecast_seasonal_naive[idx-1]


    # PLOT DECOMPOSITION
    ##################################################
    # conceptually, this should be its own function, but it's only a few lines 
    #   of code
    ##################################################

    if plot_decomposition:
        _ = decomposition.plot()
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()

    return test_forecast_seasonal_naive


def calculate_forecasts_and_metrics(
    time_series: np.ndarray, test_start_idx: int, 
    model_result: sarimax.SARIMAXResultsWrapper, period: int, 
    decompose_additive: bool, plot_decomposition: bool=False,
    decomposition_plot_filepath: Path=Path('plot.png')
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce naive, seasonal naive, and model forecasts and calculate metrics
        for those forecasts and for the training portion of the time series
    Seasonal naive forecast is produced by decomposing the time series into 
        trend and seasonal components

    time_series - a 1-dimensional array representing the time series
    model_result - results from fitted 'statsmodels' SARIMAX model
    test_start_idx - the index of the time series at which the test set starts
    period - the period of the seasonal component
    decompose_additive - whether to decompose the time series using an additive 
        model ('True') or a multiplicative model ('False')
    plot_decomposition - whether to save a plot of the decomposition
    output_filepath - the filepath at which to save the decomposition plot
    """

    # INPUT PRE-CHECKS AND SETTINGS
    ##################################################

    assert is_array_one_dimensional(time_series)
    assert test_start_idx > 0
    assert test_start_idx < len(time_series)

    assert period < len(time_series) // 2


    # DERIVE VARIABLES FROM PARAMETERS
    ##################################################

    train_series = time_series[:test_start_idx]
    test_series = time_series[test_start_idx:]

    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)
    fittedvalues = model_result.fittedvalues


    # CALCULATE FORECASTS
    ##################################################

    test_forecast_model = model_result.forecast(steps=len(test_series))
    test_forecast_naive = np.repeat(fittedvalues[-1], len(test_series))
    test_forecast_seasonal_naive = decompose_and_forecast_seasonal_naive(
        time_series, test_start_idx, fittedvalues[-1], period, 
        decompose_additive, plot_decomposition, decomposition_plot_filepath)

    forecast_df = pd.DataFrame({
        'naive_forecast': test_forecast_naive,
        'test_forecast_seasonal_naive': test_forecast_seasonal_naive,
        'model_forecast': test_forecast_model})


    # CALCULATE FORECAST METRICS
    ##################################################

    train_metrics = calculate_time_series_metrics(train_series, fittedvalues)
    test_metrics = calculate_time_series_metrics(test_series, test_forecast_model)
    test_metrics_naive = calculate_time_series_metrics(
        test_series, test_forecast_naive)
    test_metrics_seasonal_naive = calculate_time_series_metrics(
        test_series, test_forecast_seasonal_naive)


    # COMPILE FORECAST METRICS
    ##################################################
    # convert metrics dataclasses to dataframe
    ##################################################

    metrics = [
        train_metrics, test_metrics, test_metrics_naive, 
        test_metrics_seasonal_naive]

    metrics_dict = {}
    for metrics_class in metrics:
        for field in fields(metrics_class):
            field_value = getattr(metrics_class, field.name)
            if field.name in metrics_dict:
                metrics_dict[field.name].append(field_value)
            else:
                metrics_dict[field.name] = [field_value]

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.index = ['train', 'test', 'test_naive', 'test_seasonal_naive']

    return forecast_df, metrics_df


##################################################
# TIME SERIES PLOTS
##################################################

def plot_time_series(
    time_series: np.ndarray, series_n_to_plot: int=1, title: str='Time Series',
    output_filepath: Path=Path('plot.png')) -> None:

    assert series_n_to_plot <= time_series.shape[0]

    # plt.figure(figsize=(12, 3))
    for srs in time_series[:series_n_to_plot]:
        plt.plot(srs, alpha=0.5)

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig(output_filepath)

    plt.clf()
    plt.close()


def plot_time_series_autocorrelation(
    ts: list[np.ndarray], output_filepath: Path=Path('plot.png')) -> None:
    """

    Adapted from:
        https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    """

    plt.rcParams.update({'figure.figsize': (16, 3*len(ts))})

    fig, axes = plt.subplots(len(ts), 3, sharex=False)

    for i, _ in enumerate(ts):
        axes[i, 0].plot(ts[i]); axes[i, 0].set_title(f'Series #{i}')
        tsa_plots.plot_acf(ts[i], ax=axes[i, 1])
        tsa_plots.plot_pacf(ts[i], ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_time_series_and_model_values_1(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')) -> None:

    train_steps_n = len(model_result.fittedvalues)
    forecast_steps_n = len(original_series) - train_steps_n
    forecast = model_result.forecast(steps=forecast_steps_n)

    plt.plot(original_series, alpha=0.5, color='blue')
    plt.plot(model_result.fittedvalues, alpha=0.5, color='green')

    forecast_idx = range(train_steps_n, train_steps_n + forecast_steps_n)
    plt.plot(forecast_idx, forecast, alpha=0.5, color='orange')

    plt.title('Original series (blue), fitted values (green), forecast (orange)')
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_time_series_and_model_values_2(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')) -> None:

    prediction = model_result.predict(start=0, end=len(original_series))

    plt.plot(original_series, alpha=0.5, color='blue')
    plt.plot(prediction, alpha=0.5, color='green')

    plt.title(
        'Original series (blue), model values (green), fit vs. forecast (red)')
    plt.axvline(x=len(model_result.fittedvalues), color='red', linestyle='--')
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_time_series_and_model_values_3(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')) -> None:

    simulations = model_result.simulate(
        nsimulations=len(original_series), repetitions=50).squeeze()

    plt.plot(simulations, alpha=0.1, color='green')
    plt.plot(original_series, alpha=0.9, color='blue')

    plt.title(
        'Original series (blue), simulated values based on model (green)')
    plt.axvline(x=len(model_result.fittedvalues), color='red', linestyle='--')
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
