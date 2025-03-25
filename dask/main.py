#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0

import logging
import dask.distributed
import module
import module_dask

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    logging.basicConfig(filename="time_log.log", level=logging.INFO, format="%(asctime)s - %(message)s")

    df_clin, df_expression = module.load_data_pd(log=logging)
    df_clin = module.initial_preprocessing_pd(logging, df_clin)
    df_combined, grouped_df = module.data_exploration_pd(logging, df_clin, df_expression)
    module.modelling_XGBoost_pd(logging, df_combined)

    df_clin, df_expression = module_dask.load_data_dask(log=logging)
    df_clin = module_dask.initial_preprocessing_dask(logging, df_clin)
    df_combined, _ = module_dask.data_exploration_dask(logging, df_clin, df_expression)

    client = dask.distributed.Client()
    module_dask.modelling_XGBoost_dask(logging, df_combined, client)
    client.close()

if __name__ == '__main__':
    main()