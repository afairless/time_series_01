 
import pytest
import numpy as np

from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st

from sklearn.metrics import (
    root_mean_squared_error as skl_rmse, 
    mean_absolute_error as skl_mae, 
    median_absolute_error as skl_mdae)

from skforecast.metrics.metrics import (
    mean_squared_error as skf_mse,
    mean_absolute_error as skf_mae,
    median_absolute_error as skf_mdae)

from pmdarima.utils import diff as pm_diff
from pmdarima.utils import diff_inv as pm_diff_inv
import statsmodels.tsa.statespace.sarimax as sarimax

from src.common import (
    is_array_one_dimensional,
    root_median_squared_error,
    TimeSeriesDifferencing,
    )


def test_is_array_one_dimensional_01():
    """
    Test valid input
    """
    arr = np.array([1])
    result = is_array_one_dimensional(arr)
    assert result == True


def test_is_array_one_dimensional_02():
    """
    Test valid input
    """
    arr = np.array([1, 2, 3, 4, 5])
    result = is_array_one_dimensional(arr)
    assert result == True


def test_is_array_one_dimensional_03():
    """
    Test valid input
    """
    arr = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    result = is_array_one_dimensional(arr)
    assert result == True


def test_is_array_one_dimensional_04():
    """
    Test valid input
    """
    arr = np.array([1, 2, 3, 4, 5]).reshape(-1, 1, 1)
    result = is_array_one_dimensional(arr)
    assert result == True


def test_is_array_one_dimensional_05():
    """
    Test valid input
    """
    arr = np.array([1, 2, 3, 4, 5]).reshape(-1, 1, 1, 1)
    result = is_array_one_dimensional(arr)
    assert result == True


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_mean_squared_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_rmse(y1, y2)
    skf_result = np.sqrt(skf_mse(y1, y2))
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_mean_absolute_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_mae(y1, y2)
    skf_result = skf_mae(y1, y2)
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_median_absolute_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_mdae(y1, y2)
    skf_result = skf_mdae(y1, y2)
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_median_squared_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = y1
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([0]))


def test_median_squared_error_02():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 2, 1])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([0]))


def test_median_squared_error_03():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 1, 1])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([1]))


def test_median_squared_error_04():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([2]))


def test_median_squared_error_05():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3, 4])
    y2 = np.array([0, 0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([np.sqrt(13/2)]))


def test_median_squared_error_06():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([0, 0, 0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([3]))


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(time_series, k_diff=1, k_seasonal_diff=0)
    ts_diff_1 = time_series[1:] - time_series[:-1]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - 1


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    period=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_02(
    low: int, high: int, arr_len: int, period: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=0, k_seasonal_diff=1, seasonal_periods=period)
    ts_diff_1 = time_series[period:] - time_series[:-period]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - period


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    period=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_03(
    low: int, high: int, arr_len: int, period: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=1, k_seasonal_diff=1, seasonal_periods=period)

    ts_diff_1 = time_series[period:] - time_series[:-period]
    ts_diff_2 = ts_diff_1[1:] - ts_diff_1[:-1]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_2)
    assert len(ts_diff_0) == len(time_series) - (period+1)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    period=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_04(
    low: int, high: int, arr_len: int, period: int, seed: int):
    """
    Verify that sarimax differencing and pmdarima differencing are equivalent
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=1, k_seasonal_diff=1, seasonal_periods=period)

    ts_diff_1 = pm_diff(time_series, lag=period, differences=1)
    ts_diff_2 = pm_diff(ts_diff_1, lag=1, differences=1)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_2)
    assert len(ts_diff_0) == len(time_series) - (period+1)


def test_difference_time_series_seasonal_01():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 0
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = time_series

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_02():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([6, 7, -2, -3, -2, -2])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_03():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 2
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([-8, -10, 0, 1])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_04():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 3
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([8, 11])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_05():
    """
    Test seasonal differencing only
    k_seasonal_diff * seasonal_periods uses all of input array
    """

    k_seasonal_diff = 4
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_06():
    """
    Test seasonal differencing only
    k_seasonal_diff * seasonal_periods exceeds length of input array
    """

    k_seasonal_diff = 5
    seasonal_periods = 2
    time_series = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_07():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 1
    seasonal_periods = 3
    time_series = np.array([3, 5, 1, 8, 2, 0, 7, 9, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([5, -3, -1, -1, 7, 4])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_08():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 2
    seasonal_periods = 3
    time_series = np.array([3, 5, 1, 8, 2, 0, 7, 9, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([-6, 10, 5])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_09():
    """
    Test seasonal differencing only
    k_seasonal_diff * seasonal_periods uses all of input array
    """

    k_seasonal_diff = 3
    seasonal_periods = 3
    time_series = np.array([3, 5, 1, 8, 2, 0, 7, 9, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_10():
    """
    Test seasonal differencing only
    k_seasonal_diff * seasonal_periods exceeds length of input array
    """

    k_seasonal_diff = 4
    seasonal_periods = 3
    time_series = np.array([3, 5, 1, 8, 2, 0, 7, 9, 4])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_11():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series = np.array([0.51182162, 0.9504637 , 0.14415961, 0.94864945])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([-0.36766201, -0.00181425])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_seasonal_12():
    """
    Test seasonal differencing only
    """

    k_seasonal_diff = 1
    seasonal_periods = 1
    time_series = np.array([0.51182162, 0.9504637 , 0.14415961, 0.94864945])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.difference_time_series_seasonal(
        time_series, k_seasonal_diff, seasonal_periods)

    correct_result = np.array([ 0.43864208, -0.80630409,  0.80448984])

    np.testing.assert_almost_equal(result, correct_result)


def test_difference_time_series_01():
    """
    Test invalid input:  array with >1 dimension
    """

    time_series = np.array([[1, 2, 3], [4, 5, 6]])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    with pytest.raises(AssertionError):
        _ = ts_diff.difference_time_series(time_series)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_difference_time_series_02(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test no differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    ts_diff_1 = ts_diff.difference_time_series(time_series)
    np.testing.assert_almost_equal(time_series, ts_diff_1)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=4, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4))
@settings(print_blob=True)
def test_difference_time_series_03(
    low: int, high: int, arr_len: int, seed: int, k_diff: int):
    """
    Test simple differencing only
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(time_series, k_diff=k_diff)

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - k_diff


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=50))
@settings(print_blob=True)
def test_difference_time_series_04(
    low: int, high: int, arr_len: int, seed: int, k_seasonal_diff: int, 
    seasonal_periods: int):
    """
    Test seasonal differencing only
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=0, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)

    ts_diff = TimeSeriesDifferencing(
        k_diff=0, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    assert len(ts_diff_0) == max(0, (len(time_series) - season_period_diff_len))


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len_factor=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=50))
@settings(print_blob=True)
def test_difference_time_series_05(
    low: int, high: int, arr_len_factor: int, seed: int, k_diff: int, 
    k_seasonal_diff: int, seasonal_periods: int):
    """
    Test simple and seasonal differencing against statsmodels sarimax 'diff'
    """

    rng = np.random.default_rng(seed)
    low = k_diff
    high = max(low+1, seasonal_periods)
    simple_len = rng.integers(low=low, high=high, size=1)
    arr_len = (k_seasonal_diff * seasonal_periods * arr_len_factor) + simple_len
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)
    ts_diff_2 = ts_diff._difference_time_series_2(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_2)
    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    diff_total = k_diff + season_period_diff_len 
    assert len(ts_diff_0) == max(0, (len(time_series) - diff_total))


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len_factor=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=50))
@settings(print_blob=True)
def test_difference_time_series_06(
    low: int, high: int, arr_len_factor: int, seed: int, k_diff: int, 
    k_seasonal_diff: int, seasonal_periods: int):
    """
    Test simple and seasonal differencing against pmdarima 'diff'
    """

    rng = np.random.default_rng(seed)
    low = k_diff
    high = max(low+1, seasonal_periods)
    simple_len = rng.integers(low=low, high=high, size=1)
    arr_len = (k_seasonal_diff * seasonal_periods * arr_len_factor) + simple_len
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0a = pm_diff(
        time_series, lag=seasonal_periods, differences=k_seasonal_diff)
    ts_diff_0b = pm_diff(ts_diff_0a, lag=1, differences=k_diff)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)
    ts_diff_2 = ts_diff._difference_time_series_2(time_series)

    np.testing.assert_almost_equal(ts_diff_0b, ts_diff_2)
    np.testing.assert_almost_equal(ts_diff_0b, ts_diff_1)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    diff_total = k_diff + season_period_diff_len 
    assert len(ts_diff_0b) == max(0, (len(time_series) - diff_total))


def test_difference_time_series_07():
    """
    Test simple and seasonal differencing
    """

    k_diff = 1
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series = np.array([3, 5, 2, 6])

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)
    ts_diff_2 = ts_diff._difference_time_series_2(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    np.testing.assert_almost_equal(ts_diff_0, ts_diff_2)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    diff_total = k_diff + season_period_diff_len 
    assert len(ts_diff_0) == max(0, (len(time_series) - diff_total))


def test_periodic_cumulative_sum_01():
    """
    Test invalid input:  empty input array
    """

    seasonal_periods = 2
    series = np.array([])

    ts_diff = TimeSeriesDifferencing()

    with pytest.raises(AssertionError):
        _ = ts_diff.periodic_cumulative_sum(series, seasonal_periods)


def test_periodic_cumulative_sum_02():
    """
    Test invalid input:  array with >1 dimension
    """

    seasonal_periods = 2
    series = np.array([[1, 2], [3, 4]])

    ts_diff = TimeSeriesDifferencing()

    with pytest.raises(AssertionError):
        _ = ts_diff.periodic_cumulative_sum(series, seasonal_periods)


def test_periodic_cumulative_sum_03():
    """
    Test valid input
    """

    seasonal_periods = 3
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([1, 2, 3, 5, 7, 9, 12, 15, 18])

    np.testing.assert_almost_equal(correct_result, result)


def test_periodic_cumulative_sum_04():
    """
    Test valid input
    """

    seasonal_periods = 2
    series = np.array([1, 2, 7, 8, -1, -2, -1, -1])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([1, 2, 8, 10, 7, 8, 6, 7])

    np.testing.assert_almost_equal(correct_result, result)


def test_periodic_cumulative_sum_05():
    """
    Test valid input:  length of input 'series' is not a multiple of 
        'seasonal_periods '
    """

    seasonal_periods = 2
    series = np.array([1, 2, -2, 3, -5])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([1, 2, -1, 5, -6])

    np.testing.assert_almost_equal(correct_result, result)


def test_periodic_cumulative_sum_06():
    """
    Test valid input:  length of input 'series' is not a multiple of 
        'seasonal_periods '
    """

    seasonal_periods = 2
    series = np.array([1, 2, 3, 4, 5])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([1, 2, 4, 6, 9])

    np.testing.assert_almost_equal(correct_result, result)


def test_periodic_cumulative_sum_07():
    """
    Test valid input:  length of input 'series' is not a multiple of 
        'seasonal_periods '
    """

    seasonal_periods = 3
    series = np.array([5, -3, 1, 2, 8, 0, -2, -5])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([5, -3, 1, 7, 5, 1, 5, 0])

    np.testing.assert_almost_equal(correct_result, result)


def test_periodic_cumulative_sum_08():
    """
    Test valid input:  length of input 'series' is not a multiple of 
        'seasonal_periods '
    """

    seasonal_periods = 2
    series = np.array([-3, 4, 8, -8, -6, 0, 1, 9, -2])

    ts_diff = TimeSeriesDifferencing()
    result = ts_diff.periodic_cumulative_sum(series, seasonal_periods)

    correct_result = np.array([-3, 4, 5, -4, -1, -4, 0, 5, -2])

    np.testing.assert_almost_equal(correct_result, result)


def test_de_difference_time_series_01():
    """
    Test invalid input:  no differencing before de-differencing
    """

    time_series = np.array([1, 2, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    with pytest.raises(ValueError):
        _ = ts_diff.de_difference_time_series(time_series)


def test_de_difference_time_series_02():
    """
    Test invalid input:  array with >1 dimension
    """

    time_series_1 = np.array([1, 2, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(AssertionError):
        _ = ts_diff.de_difference_time_series(time_series_2)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_03(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test no differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    _ = ts_diff.difference_time_series(time_series)

    ts_diff_2 = ts_diff.de_difference_time_series(time_series)
    np.testing.assert_almost_equal(time_series, ts_diff_2)


def test_de_difference_time_series_04():
    """
    Test simple differencing only:  empty de-differencing input vector
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 0, 4, -8, -3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    ts_diff_2 = ts_diff.de_difference_time_series()
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_05():
    """
    Test simple differencing only:  empty de-differencing input vector
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 0, 4, -8, -3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([])

    ts_diff_2 = ts_diff.de_difference_time_series(time_series_2)
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_06():
    """
    Test simple differencing only:  k_diff = 1
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 5, 10, 8])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1, 1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([1, 3, 7, 13, 12])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_07():
    """
    Test simple differencing only:  k_diff = 2
    """

    k_diff = 2
    time_series_1 = np.array([7, 3, 9, 1, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([7, 3, 10, 4, 9])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_08():
    """
    Test simple differencing only:  k_diff = 3
    """

    k_diff = 3
    time_series_1 = np.array([2, 8, 7, 2, 4])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([2, 8, 7, 3, 8])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_09():
    """
    Test simple differencing only:  invert differencing, k_diff = 1
    """

    time_series = np.array([2, 1, 3, 5])

    ts_diff = TimeSeriesDifferencing(k_diff=1)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    np.testing.assert_almost_equal(time_series, ts_diff_1)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=4, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4))
@settings(print_blob=True)
def test_de_difference_time_series_10(
    low: int, high: int, arr_len: int, seed: int, k_diff: int):
    """
    Test simple differencing only:  invert differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=3)


def test_de_difference_time_series_11():
    """
    Test seasonal differencing only:  
        k_seasonal_diff = 1
        empty de-differencing input vector
        seasonal_periods = 2
    """

    k_diff = 0
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series_1 = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    ts_diff_2 = ts_diff.de_difference_time_series()
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_12():
    """
    Test seasonal differencing only:  
        k_seasonal_diff = 1
        empty de-differencing input vector
        seasonal_periods = 2
    """

    k_diff = 0
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series_1 = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([])

    ts_diff_2 = ts_diff.de_difference_time_series(time_series_2)
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_13():
    """
    Test seasonal differencing only:  
        k_seasonal_diff = 1
        seasonal_periods = 2
    """

    k_diff = 0
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series_1 = np.array([1, 2, 7, 9, 5, 6, 3, 4])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1, 1, 1, 1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([1, 2, 8, 10, 7, 8, 6, 7])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_14():
    """
    Test seasonal differencing only:  
        k_seasonal_diff = 2
        seasonal_periods = 3
    """

    k_diff = 0
    k_seasonal_diff = 2
    seasonal_periods = 3
    time_series_1 = np.array([1, 2, 7, 9, 5, 6, 3, 4, 0])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([4, 3, 2])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([1, 2, 7, 9, 5, 6, 7, 7, 2])

    np.testing.assert_almost_equal(result, correct_result)


@given(
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=12),
    arr_len_factor=st.integers(min_value=1, max_value=20),
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_15(
    k_seasonal_diff: int, seasonal_periods: int, arr_len_factor: int, low: int, 
    high: int, seed: int):
    """
    Test seasonal differencing only:  invert differencing
    """

    k_diff = 0
    arr_len = k_seasonal_diff * seasonal_periods * arr_len_factor
    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=3)


def test_de_difference_time_series_16():
    """
    Test simple and seasonal differencing:  invert differencing
    """

    k_diff = 1
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series = np.array([3, 5, 2, 6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=3)


def test_de_difference_time_series_17():
    """
    Test simple and seasonal differencing
    """

    k_diff = 1
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series_1 = np.array([3, 5, 2, 6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([3, 5, 2, 7])

    np.testing.assert_almost_equal(result, correct_result)


@given(
    k_diff=st.integers(min_value=1, max_value=4),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=12),
    arr_len_factor=st.integers(min_value=1, max_value=20),
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_18(
    k_diff: int, k_seasonal_diff: int, seasonal_periods: int, 
    arr_len_factor: int, low: int, high: int, seed: int):
    """
    Test simple and seasonal differencing:  invert differencing
    """

    rng = np.random.default_rng(seed)
    low = k_diff
    high = max(low+1, seasonal_periods)
    simple_len = rng.integers(low=low, high=high, size=1)
    arr_len = (k_seasonal_diff * seasonal_periods * arr_len_factor) + simple_len
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=0)


def test_de_difference_time_series_19():
    """
    Test simple and seasonal differencing
    """

    k_diff = 1
    k_seasonal_diff = 1
    seasonal_periods = 2
    time_series_1 = np.array([1, 2, -1, 5, -6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([-3, 4])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([1, 2, -1, 2, -5])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_20():
    """
    Test simple and seasonal differencing
    """

    k_diff = 2
    k_seasonal_diff = 1
    seasonal_periods = 4
    time_series_1 = np.array([3, -1, -6, 3, 4, -2, 0, 1])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([-4, 11])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([3, -1, -6, 3, 4, -2, -4, 4])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_21():
    """
    Test simple and seasonal differencing
    """

    k_diff = 1
    k_seasonal_diff = 2
    seasonal_periods = 3
    time_series_1 = np.array([9, 2, -2, -5, 1, 0, 7, 0])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([28])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([9, 2, -2, -5, 1, 0, 7, 28])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_22a():
    """
    Test simple and seasonal differencing:  seasonal differencing with 
        'seasonal_periods' = 1 is same as simple differencing, example #1
    """

    k_diff = 2
    k_seasonal_diff = 0
    seasonal_periods = 1
    time_series_1 = np.array([-4, -3, 5, 4, 6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([-5, 6, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([-4, -3, 0, 0, 4])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_22b():
    """
    Test simple and seasonal differencing:  seasonal differencing with 
        'seasonal_periods' = 1 is same as simple differencing, example #1
    """

    k_diff = 1
    k_seasonal_diff = 1
    seasonal_periods = 1
    time_series_1 = np.array([-4, -3, 5, 4, 6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([-5, 6, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([-4, -3, 0, 0, 4])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_22c():
    """
    Test simple and seasonal differencing:  seasonal differencing with 
        'seasonal_periods' = 1 is same as simple differencing, example #1
    """

    k_diff = 0
    k_seasonal_diff = 2
    seasonal_periods = 1
    time_series_1 = np.array([-4, -3, 5, 4, 6])

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([-5, 6, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([-4, -3, 0, 0, 4])

    np.testing.assert_almost_equal(result, correct_result)


@given(
    k_diff=st.integers(min_value=1, max_value=4),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=12),
    arr_len_factor=st.integers(min_value=1, max_value=10),
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_23(
    k_diff: int, k_seasonal_diff: int, seasonal_periods: int, 
    arr_len_factor: int, low: int, high: int, seed: int):
    """
    Test simple and seasonal differencing:  invert differencing
    """

    rng = np.random.default_rng(seed)
    low = k_diff
    high = max(low+1, seasonal_periods)
    simple_len = rng.integers(low=low, high=high, size=1)
    arr_len = (k_seasonal_diff * seasonal_periods * arr_len_factor) + simple_len
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    ts_diff_2a = pm_diff(
        time_series, lag=seasonal_periods, differences=k_seasonal_diff)
    ts_diff_2b = pm_diff(ts_diff_2a, lag=1, differences=k_diff)

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=1)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=4, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_23(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test simple differencing only:  invert differencing
    Test inverse differencing against pmdarima 'diff_inv', but only for 
        k_diff = 1, because parameter 'xi' seems inoperable and I don't want to 
        re-code the de-differencing from the prepend vector as in 
        'TimeSeriesDifferencing.de_difference_time_series'
    """

    k_diff = 1
    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    # pmdarima 'diff_inv' won't accept empty arrays, i.e., arrays of length 0
    if len(time_series) == k_diff:
        k_diff = k_diff - 1

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    ts_diff_0 = ts_diff.difference_time_series(time_series)
    ts_diff_1 = ts_diff.de_difference_time_series(ts_diff_0)

    # parameter 'xi' of pmdarima 'diff_inv' seems not to be functional
    ts_diff_2 = pm_diff(time_series, lag=1, differences=k_diff)
    ts_diff_3a = pm_diff_inv(ts_diff_2, lag=1, differences=k_diff)
    ts_diff_3b = ts_diff_3a + time_series[:k_diff]

    np.testing.assert_almost_equal(time_series, ts_diff_1, decimal=3)
    np.testing.assert_almost_equal(ts_diff_1, ts_diff_3b, decimal=3)


