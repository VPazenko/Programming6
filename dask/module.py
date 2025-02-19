#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0


import dask.dataframe as dd
import pandas as pd
import dask
import time
import gzip
import os


def load_data_pd(log, metadata='data/Lung3.metadata.xlsx', expression_data="data/GSE58661_series_matrix.txt.gz"):
    # Start timer
    start = time.time()
    # Load metadata
    df_clin = pd.read_excel(metadata)
    # Load expression data
    headers = pd.read_csv(expression_data, compression="gzip", header=None, skiprows=28, nrows=1, sep='\t').values.flatten()
    df_exp = pd.read_csv(expression_data, compression="gzip", skiprows=63, names=headers, sep="\t")
    # Memory Usage
    memory_clin = df_clin.memory_usage(deep=True).sum() / (1024**2)  # Convert to MB
    memory_exp = df_exp.memory_usage(deep=True).sum() / (1024**2)  # Convert to MB
    # End timer
    pandas_time = time.time() - start

    log.info(f"1. Pandas load time: {pandas_time:.2f} seconds")
    log.info(f"1. Pandas memory usage (clinical): {memory_clin:.2f} MB")
    log.info(f"1. Pandas memory usage (expression): {memory_exp:.2f} MB")

    return df_clin, df_exp




