#!/usr/bin/env python3
"""
Module contain function that add a BioGrid 'Score' column to your df 
You have to provide df with at least 2 columns (Gene_A and Gene_B)
You can specify column names in function
"""

__author__ = "V.Pazenko"

# imports
import pandas as pd

def add_BioGrid_score_column(eQTL, cis_gene='cis_eQTL_gene_name', trans_gene='trans_eQTL_gene_name',\
                             biogrid_db_dir="../../Data/BioGrid/biogrid.csv"):
    """
    Input: 1. eQTL - initial data (in pandas.df format)
           2. cis_gene - specify column in eQTL df that contain cis-eQTL gene name
           3. trans_gene - specify column in eQTL df that contain trans-eQTL gene name
           4. biogrid_db_dir - specify a directory for the BioGrid db

    Function add new column (BioGrid_Score) to original df 

    Output: result - modified df (original one with new column)
    """
    # Download biogrid database
    bio_grid = pd.read_csv(biogrid_db_dir)

    # Just to be safe, create a new columns with all upper cases
    eQTL['geneA'] = eQTL[cis_gene].fillna(0).astype(str).apply(lambda x: x.upper())
    eQTL['geneB'] = eQTL[trans_gene].fillna(0).astype(str).apply(lambda x: x.upper())

    bio_grid['A'] = bio_grid.iloc[:,0].fillna(0).astype(str).apply(lambda x: x.upper())
    bio_grid['B'] = bio_grid.iloc[:,1].fillna(0).astype(str).apply(lambda x: x.upper())

    # Combine 2 dataframes (and drop all duplicates)
    eQTL_combined = eQTL.merge(bio_grid.drop_duplicates(),\
                                    left_on=["geneA", "geneB"], right_on=["A", "B"], how="left")
    # Insert new column into original df
    new_column = eQTL_combined['Score']
    result = eQTL.merge(new_column.rename('BioGrid_Score'), left_index=True, right_index=True)

    return result


eQTL = pd.read_csv(snakemake.input[0], sep="\t")

eqtl_biogrid = add_BioGrid_score_column(eQTL, biogrid_db_dir=snakemake.input[1])
eqtl_biogrid.to_csv(snakemake.output[0], index=False)