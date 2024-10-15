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


def main():

    input_path = Path.cwd() / 'input'
    input_filepath = input_path / 'uschange.rda'
    df = pyreadr.read_r(input_filepath)


if __name__ == '__main__':
    main()
