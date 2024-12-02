#!/usr/bin/env python3
"""
Module contain function that add a ReMap interactions column to your df 
You have to provide df with at least 2 columns (Gene_A and Gene_B)
You can specify column names in function and remap_info file directory
"""

__author__ = "V.Pazenko"

# imports
import pandas as pd
import numpy as np
import yaml


def add_ReMap_column(eQTL, cis_gene='cis_eQTL_gene_name', trans_gene='trans_eQTL_gene_name',\
                            remap_db_dir="../../Data/transcription_factor_gene_bindings.csv"):
    """
    Input: 1. eQTL - initial data (in pandas.df format)
           2. cis_gene - specify column in eQTL df that contain cis-eQTL gene name
           3. trans_gene - specify column in eQTL df that contain trans-eQTL gene name
           4. remap_db_dir - specify a directory for the ReMap info db

    Function add new column (ReMap) with True/False statement to original df 

    Output: eQTL_combined - modified df (original one with new column)
    """
    # Download remap database
    remap_info = pd.read_csv(remap_db_dir)
    # clear TFs names (delete everything after ':')
    remap_info['transcription_factor'] = remap_info['transcription_factor'].str.split(':').str[0]

    # Combine 2 dataframes (and drop all duplicates)
    eQTL_combined = eQTL.merge(remap_info.drop_duplicates(), left_on=[cis_gene, trans_gene],\
                               right_on=["transcription_factor", "Gene"], how="left")
    # This should release RAM
    del remap_info
    # Fill in all empty values (where remap info not present)
    eQTL_combined.Gene = eQTL_combined.Gene.fillna('False')
    # Create new column
    eQTL_combined['ReMap'] = np.where(eQTL_combined['Gene']=='False', False, True)
    # Drop other column from remap
    eQTL_combined = eQTL_combined.drop(columns=['transcription_factor', 'Gene'])

    return eQTL_combined


# with open("../../config/config.yaml", "r") as config_file:
#     config = yaml.safe_load(config_file)

eQTL = pd.read_csv(snakemake.input[0], sep="\t")
eqtl_remap = add_ReMap_column(eQTL, remap_db_dir=snakemake.input[1])
eqtl_remap.to_csv(snakemake.output[0], index=False)
