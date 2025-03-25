#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0

import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve


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


def data_exploration_pd(log, df_clin, df_exp):
    # Start timer
    start = time.time()
    df_exp = df_exp.set_index('!Sample_title').T

    list_std = list(df_exp.agg("std"))
    df_exp.loc['std'] = list_std
    # sort by std (max is first)
    df_exp = df_exp.sort_values(by='std', axis=1, ascending=False)

    df_clin = df_clin.set_index('title')
    df_combined = df_clin.merge(df_exp.iloc[:,0:10], how='left', left_index=True, right_index=True)
    df_combined["TumorSubtype"] = df_combined["characteristics.tag.histology"].str.contains("Squamous", case=False, na=False).astype(int)
    del df_combined["characteristics.tag.histology"]

    # I suppose, that we can decrease number of variables in the primary stage column (because subtypes pt1a, pt1b, pt1c are similar)
    df_combined['characteristics.tag.stage.primary.tumor'] = df_combined['characteristics.tag.stage.primary.tumor'].apply(lambda x: x[:3])

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


def preprocessing_train_test_pd(log, df_combined):
    # Start timer
    start = time.time()

    X = df_combined.iloc[:,:-1]
    cat_columns = df_combined.iloc[:,:-1].select_dtypes(include=['object']).columns
    num_columns = df_combined.iloc[:,:-1].select_dtypes(include=['number']).columns
    X = pd.get_dummies(X, columns = cat_columns, drop_first=True)
    y = df_combined.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Do not leak data from test to train
    mean = X_train[num_columns].mean()
    std = X_train[num_columns].std()

    # use z-score
    X_train[num_columns] = (X_train[num_columns] - mean) / std
    X_test[num_columns] = (X_test[num_columns] - mean) / std

    # End timer
    preprocess_time = time.time() - start

    log.info(f"4. Pandas preprocessing time: {preprocess_time:.2f} seconds")

    return X_train, X_test, y_train, y_test


def modelling_XGBoost_pd(X_train, X_test, y_train, y_test):
    # start timer
    start_time = time.time()
    # Convert Pandas data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss', 
        'use_label_encoder': False
    }
    # Train the model
    model_pandas = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

    # Predict on the test set
    y_pred_pandas = model_pandas.predict(dtest)
    y_pred_binary_pandas = (y_pred_pandas > 0.5).astype(int)

    # Calculate metrics
    accuracy_pandas = accuracy_score(y_test, y_pred_binary_pandas)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_binary_pandas, average='binary')
    auc_roc_pandas = roc_auc_score(y_test, y_pred_pandas)

    end_time = time.time()
    print(f"Training time (Pandas): {end_time - start_time:.2f} seconds")

    # Print metrics for Pandas
    print(f"Accuracy: {accuracy_pandas:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc_pandas:.2f}")
    roc_curve(y_test, y_pred_pandas, auc_roc_pandas, name='pandas')


def roc_curve(y_test, y_pred, auc_score, name='pandas'):

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 50% probability line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({name})")
    plt.legend()

    plt.savefig(f"plots/ROC_curve_{name}.png")
