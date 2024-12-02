#!/usr/bin/env python3

__author__ = "V.Pazenko"

# imports
import pandas as pd
import numpy as np

import logging
# Настройка логирования
logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG)

# Логируем события
logging.debug("Debugging message")
logging.info("Informational message")
logging.warning("Warning message")
logging.error("Error message")


def create_df_with_thold(df_t, cis_target_dir, number_of_bp, threshold):
    file_name = cis_target_dir + 'TF_transGene' + str(number_of_bp)+ 'bp_score_above_' + str(threshold) + '.csv'
    try:
        transform_df = pd.read_csv(file_name)
        return transform_df.iloc[:,1:]
    except FileNotFoundError:
        transform_df = pd.DataFrame(columns=('TF_cluster', 'Gene_score_more_th'))
        x = 0
        for elem in list(df_t.columns):
            new_df = df_t[df_t[elem]>threshold].loc[:,elem]
            new_df = new_df.reset_index()
            new_df[elem] = elem
            new_df = new_df.rename(columns={'index': 'Gene_score_more_th', elem: 'TF_cluster'})
            transform_df = pd.concat([transform_df, new_df]).reset_index(drop=True)
            del new_df
            x += 1
            if x > 500:
                transform_df.drop_duplicates().to_csv(file_name, mode='a')
                del transform_df
                transform_df = pd.DataFrame(columns=('TF_cluster', 'Gene_score_more_th'))
                x = 0
        transform_df.drop_duplicates().to_csv(file_name, mode='a')
        del transform_df
        transform_df_full = pd.read_csv(file_name)
        return transform_df_full.iloc[:,1:]


def add_cis_target_column(eQTL, cis_gene='cis_eQTL_gene_name', trans_gene='trans_eQTL_gene_name', cis_target_dir='initial_data/', 
                          cisTarget_db_dir='hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather',\
                        motifs_db_dir='motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl.txt', threshold=50, number_of_bp=10000):
    """
    Input: 1. eQTL - initial data (in pandas.df format)
           2. cis_gene - specify column in eQTL df that contain cis-eQTL gene name
           3. trans_gene - specify column in eQTL df that contain trans-eQTL gene name
           4. cisTarget_db_dir - specify a directory for the cisTarget info db
           5. motifs_db_dir
           6. threshold

    Function add new column (cisTarget) with True/False statement to original df 

    Output: eQTL_combined - modified df (original one with new column)
    """
    # Download cisTarget database
    try:
        cis_t_info = pd.read_feather(cis_target_dir+cisTarget_db_dir)
        data = pd.read_csv(cis_target_dir+motifs_db_dir, sep="\t", header=0, dtype=str)
    except:
        cis_t_info = pd.read_feather(cisTarget_db_dir)
        data = pd.read_csv(motifs_db_dir, sep="\t", header=0, dtype=str)
    cis_t_info = cis_t_info.set_index('motifs').T

    tf_geneB_df = create_df_with_thold(cis_t_info, cis_target_dir=cis_target_dir, \
                                       number_of_bp=number_of_bp, threshold=threshold)
    del cis_t_info

    full_df = tf_geneB_df.merge(data.iloc[:,[0,5]], how='left', right_on='#motif_id', left_on='TF_cluster')
    del data
    del tf_geneB_df

    full_df = full_df.iloc[:,[3,1]]

    
    
    # Combine 2 dataframes (and drop all duplicates)
    eQTL_combined = eQTL.merge(full_df.drop_duplicates().dropna(), left_on=[cis_gene, trans_gene],\
                               right_on=["gene_name", "Gene_score_more_th"], how="left")

    del full_df
    # Fill in all empty values (where remap info not present)
    eQTL_combined.gene_name = eQTL_combined.gene_name.fillna('False')
    # Create new column
    column_name = 'CisTarget_' + str(number_of_bp) + '_bp'
    eQTL_combined[column_name] = np.where(eQTL_combined['gene_name']=='False', False, True)
    # Drop other column from remap
    eQTL_combined = eQTL_combined.drop(columns=["gene_name", "Gene_score_more_th"])

    return eQTL_combined


eQTL = pd.read_csv(snakemake.input[0], sep="\t")

eqtl_cistarget = add_cis_target_column(eQTL, cisTarget_db_dir=snakemake.input[1], 
                                       motifs_db_dir=snakemake.input[2])

eqtl_cistarget.to_csv(snakemake.output[0], index=False)
