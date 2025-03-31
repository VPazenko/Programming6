#!/usr/bin/env python3

'''
Main module for the project.
This module is responsible for loading data, preprocessing, data exploration, and modelling using both Pandas and Dask.
Put all results in the log file and create a Word document with the report.
'''

__author__ = "V.Pazenko"
__version__ = 1.0

import logging
import dask.distributed
import time
# My modules
import module
import module_dask
import docx_module
# Suppress FutureWarnings from Pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    '''
    Main function of the script.
    This function combine all other modules and call functions in the right spots.
    '''
    log_file_name = "time_log.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format="%(asctime)s - %(message)s")

    # Pandas part
    start_time = time.time()
    df_clin, df_expression = module.load_data_pd(log=logging)
    df_clin = module.initial_preprocessing_pd(logging, df_clin)
    df_combined, grouped_df = module.data_exploration_pd(logging, df_clin, df_expression)
    module.modelling_XGBoost_pd(logging, df_combined)
    logging.info("6. Total time Pandas (measured): %.2f seconds", time.time() - start_time)

    # Dask part
    start_time_dask = time.time()
    df_clin, df_expression = module_dask.load_data_dask(log=logging)
    df_clin = module_dask.initial_preprocessing_dask(logging, df_clin)
    # I use _ because we recive same df as grouped_df but in dd format
    df_combined, _ = module_dask.data_exploration_dask(logging, df_clin, df_expression)
    client = dask.distributed.Client()
    module_dask.modelling_XGBoost_dask(logging, df_combined, client)
    client.close()
    logging.info("6. Total time Dask (measured): %.2f seconds", time.time() - start_time_dask)

    # Create report part
    docx_module.create_docx_report(log_file_name, start_time, grouped_df, docx_name="Report.docx")


if __name__ == '__main__':
    main()
