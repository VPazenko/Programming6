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
import pandas as pd

from module import roc_curve, plot_values_distribution
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


def initial_preprocessing_dask(log, df_clin):
    # Start timer
    start = time.time()
       
    # Remove columns with only one unique value
    unique_counts = df_clin.nunique().compute()
    remove_list = unique_counts[unique_counts == 1].index.tolist()
    df_clin = df_clin.drop(columns=remove_list, errors='ignore')
    
    # Remove specific columns
    df_clin = df_clin.drop(columns=['sample.name', 'CEL.file'], errors='ignore')
    
    # Replace missing value in a specific row
    mean_value = df_clin[df_clin['characteristics.tag.gender'] == 'M']['characteristics.tag.tumor.size.maximumdiameter'].mean().compute()
    df_clin['characteristics.tag.tumor.size.maximumdiameter'] = df_clin['characteristics.tag.tumor.size.maximumdiameter'].fillna(round(mean_value, 2))
    
    # Convert specific columns to lowercase
    for col in ['characteristics.tag.stage.primary.tumor', 'characteristics.tag.stage.nodes', 'characteristics.tag.stage.mets']:
        df_clin[col] = df_clin[col].str.lower()
    
    # End timer
    preprocess_time = time.time() - start
    log.info(f"2. Dask preprocessing time: {preprocess_time:.2f} seconds")
    
    return df_clin


def data_exploration_dask(log, df_clin, df_exp):
    # Start timer
    start = time.time()
    
    df_exp = df_exp.set_index('!Sample_title')

    # Вычисляем стандартное отклонение всех генов
    std_series = df_exp.std(axis=1).compute()

    # Получаем топ-10 наиболее вариативных генов
    top_genes = std_series.nlargest(10).index
    df_selected = df_exp.loc[top_genes.to_list(),:]

    # Транспонируем результат (если данных не очень много)
    df_transposed = df_selected.compute().T  # Транспонируем в Pandas
    df_transposed = dd.from_pandas(df_transposed, npartitions=4)  # Конвертируем обратно в Dask

    df_clin = df_clin.set_index('title')
    df_combined = df_clin.merge(df_transposed, how='left', left_index=True, right_index=True)
    
    # Create TumorSubtype column
    df_combined['TumorSubtype'] = df_combined['characteristics.tag.histology'].str.contains("Squamous", case=False, na=False).astype(int)
    df_combined = df_combined.drop(columns=['characteristics.tag.histology'], errors='ignore')
    
    # Reduce number of primary stage categories
    df_combined['characteristics.tag.stage.primary.tumor'] = df_combined['characteristics.tag.stage.primary.tumor'].str[:3]
    
    # Compute grouped mean values
    last_10_columns = df_combined.columns[-11:-1]
    grouped_df = df_combined.groupby("TumorSubtype")[last_10_columns].mean().reset_index()
    
    # Plot distribution with reduced memory usage
    plot_values_distribution(df_combined.loc[:,["TumorSubtype"]].compute(), "TumorSubtype", name='dask')
    
    # End timer
    exploration_time = time.time() - start
    log.info(f"3. Dask data exploration time: {exploration_time:.2f} seconds")
    
    return df_combined, grouped_df


