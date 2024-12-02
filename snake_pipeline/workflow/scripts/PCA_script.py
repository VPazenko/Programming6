#!/usr/bin/env python3
"""
This module loads data and combines it with an eQTL datatable and saves it as a csv file.
This module requires the following files:
* 
"""
__author__ = "Gijs Bakker (modified for snakemake by V.Pazenko)"
__version__ = 1.0

import pandas as pd


def combine_components_difference(df: pd.DataFrame, components: pd.DataFrame):
    # left merge on cis gene, left merge on trans gene
    df = df.merge(components, how="left", left_on="cis_eQTL_gene", right_on="-")
    df = df.merge(components, how="left", left_on="trans_eQTL_gene", right_on="-")
    
    # difference compxA and compxB
    for i in range(1, 101):
        df = df.assign(**{f"comp{i}":df[f"Comp{i}_x"] - df[f"Comp{i}_y"]})

    # drop
    to_drop = [f"Comp{i}_{j}" for i in range(1,101) for j in "xy"] + ["-_x", "-_y"]
    df = df.drop(to_drop, axis=1)
    return df


eQTL = pd.read_csv(snakemake.input[0], sep="\t")

eigenvectors = pd.read_csv(snakemake.input[1], sep="\t")

eqtl_PCA = combine_components_difference(eQTL, eigenvectors)
eqtl_PCA.to_csv(snakemake.output[0], index=False)
