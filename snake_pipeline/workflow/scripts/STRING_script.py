#!/usr/bin/env python3
"""
This module loads data and combines it with an eQTL datatable and saves it as a csv file.
This module requires the following files:
* 
"""
__author__ = "Gijs Bakker (modified for snakemake by V.Pazenko)"
__version__ = 1.0

import pandas as pd


def combine_with_string(eQTL, STRING_name):
    """
    This function combines the data frame with STRING data through a left merge
    Args:
        eQTL: A pandas datafrmae containing the eQTL data
        STRING_name: A string with the path and filename of the STRING dataset
    returns:
        A pandas dataframe
    """
    STRING = pd.read_csv(STRING_name)
    # take average of duplicate scores
    STRING = STRING.groupby(["gene1", "gene2"]).mean()
    STRING = STRING.reset_index()
    # remove could not finds
    STRING[(STRING.gene1 != "Could_not_find") & (STRING.gene2 != "Could_not_find")]

    eQTL = eQTL.merge(STRING, left_on=["cis_eQTL_gene", "trans_eQTL_gene"], 
                           right_on=["gene1", "gene2"], how="left")
    eQTL = eQTL.drop(["gene1", "gene2"], axis=1)
    eQTL = eQTL.rename(columns={"combined_score":"STRING_score"})
    return eQTL


eQTL = pd.read_csv(snakemake.input[0], sep="\t")

eqtl_string = combine_with_string(eQTL, STRING_name=snakemake.input[1])
eqtl_string.to_csv(snakemake.output[0], index=False)