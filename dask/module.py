#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0


import matplotlib.pyplot as plt
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
    #headers = dd.read_csv(expression_data, compression="gzip", header=None, skiprows=28, sep='\t').head(1).values.flatten()
    #df_exp = dd.read_csv(expression_data, compression="gzip", skiprows=63, names=headers, sep="\t")
    # Загружаем заголовки (29-я строка → skiprows=28) и берем первую строку
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


def initial_preprocessing_pd(log, df_clin):
    # Start timer
    start = time.time()
    # Remove columns with only one unique value
    remove_list = []
    for name in list(df_clin.columns):
        if len(list(df_clin[name].value_counts())) == 1:
            remove_list.append(name)
    # Delete all these columns
    # print(remove_list)
    for  name in remove_list:
        del df_clin[name]
    # These columns are duplicates with 'title'
    del df_clin['sample.name']
    del df_clin['CEL.file']

    df_clin.iloc[3,4] = round(df_clin[df_clin['characteristics.tag.gender']=='M']['characteristics.tag.tumor.size.maximumdiameter'].mean(), 2)
    for col in ['characteristics.tag.stage.primary.tumor', 'characteristics.tag.stage.nodes', 'characteristics.tag.stage.mets']:
        df_clin.loc[:,col] = df_clin.loc[:,col].str.lower()
    # End timer
    preprocess_time = time.time() - start

    log.info(f"2. Pandas preprocessing time: {preprocess_time:.2f} seconds")

    return df_clin




def data_exploration_pandas(log, df_clin, df_exp):
    # Start timer
    start = time.time()
    df_exp = df_exp.set_index('!Sample_title')
    df_exp = df_exp.T
    list_std = list(df_exp.agg("std"))
    df_exp.loc['std'] = list_std
    # sort by std (max is first)
    df_exp = df_exp.sort_values(by='std', axis=1, ascending=False)

    df_clin = df_clin.set_index('title')
    df_combined = df_clin.merge(df_exp.iloc[:,0:10], how='left', left_index=True, right_index=True)
    df_combined["TumorSubtype"] = df_combined["characteristics.tag.histology"].str.contains("Squamous", case=False, na=False).astype(int)
    del df_combined["characteristics.tag.histology"]

    last_10_columns = df_combined.columns[-11:-1]
    grouped_df = df_combined.groupby("TumorSubtype")[last_10_columns].mean().reset_index()
    # grouped_df.head()

    plot_values_distribution(df_combined, "TumorSubtype", name='pandas')
    # End timer
    exploration_time = time.time() - start

    log.info(f"3. Pandas data exploration time: {exploration_time:.2f} seconds")
    return df_combined, grouped_df


def plot_values_distribution(df, column, name='pandas'):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    counts = df[column].value_counts() #.compute()
    counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column}")
    plt.xticks(rotation=0)

    plt.savefig(f"plots/{column}_distribution_{name}.png")



