#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0

import dask
import time
import gzip
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import dask.dataframe as dd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve






def load_data_dask(log, metadata='data/Lung3.metadata.xlsx', expression_data="data/GSE58661_series_matrix.txt.gz"):
    # Start timer
    start = time.time()
    # Warning gzip compression does not support breaking apart files 
    unzipped_file_path = expression_data.replace(".gz", "")
    with gzip.open(expression_data, 'rb') as f_in, open(unzipped_file_path, 'wb') as f_out:
        f_out.write(f_in.read())  # Unzip metadata
    
    end_unzip = time.time()
    # Load metadata
    delayed_df = dask.delayed(pd.read_excel)(metadata)
    df_clin = dd.from_delayed([delayed_df])
    # Load expression data
    headers = dd.read_csv(unzipped_file_path, skiprows=28, header=None, sep="\t").head(1).values.flatten()
    df_exp = dd.read_csv(unzipped_file_path, skiprows=63, names=headers, blocksize="10MB", sep="\t")
    # Memory Usage
    memory_clin = df_clin.memory_usage(deep=True).compute().sum() / (1024**2)  # Convert to MB
    memory_exp = df_exp.memory_usage(deep=True).compute().sum() / (1024**2)  # Convert to MB
    first_partition_size = df_exp.partitions[0].memory_usage(deep=True).sum().compute()/ (1024**2)

    # End timer
    dask_time = time.time() - end_unzip

    log.info(f"2. Unzip time: {end_unzip - start:.2f} seconds")
    log.info(f"2. Dask load time: {dask_time:.2f} seconds")
    log.info(f"2. Dask memory usage (clinical): {memory_clin:.2f} MB")
    log.info(f"2. Dask memory usage for the first partition (expression): {first_partition_size:.2f} MB")
    log.info(f"2. Dask memory usage aproximatle computate (expression): {memory_exp:.2f} MB")

    return df_clin, df_exp