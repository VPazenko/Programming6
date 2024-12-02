#!/usr/bin/env python3

__author__ = "V.Pazenko"

import pandas as pd
import re
from sklearn.model_selection import train_test_split


files = snakemake.input
file_to_remove = [file for file in files if re.search(r'PCA', file)]
# delete finding element
if file_to_remove:
    files.remove(file_to_remove[0])

full_data = pd.read_csv(file_to_remove[0])

for file in files:
    df = pd.read_csv(file)
    last_col = df.iloc[:, -1]
    # add last column to full_data
    full_data[df.columns[-1]] = last_col

full_data['interaction'] = full_data['posterior_prob'].apply(lambda x: 1 if pd.notna(x) else 0)

to_drop = ["cis_eQTL_gene_name", "iteration", 'type', "trans_eQTL_gene_name", "posterior_prob", "regional_prob", 
            "candidate_snp", "posterior_explained_by_snp", "dropped_trait", 'nr_snps_included','nr_trans',
            'nr_clusters', 'nr_cis_clusters', 'cis_eQTL_gene', 'trans_eQTL_gene']
full_data = full_data.drop(to_drop, axis=1)

full_data[["STRING_score", "BioGrid_Score"]] = full_data[["STRING_score", "BioGrid_Score"]].fillna(-10)

# Split the data
train_data, test_data = train_test_split(full_data, test_size=0.2)

# Save preprocessed data
train_data.to_csv(snakemake.output.train, index=False)
test_data.to_csv(snakemake.output.test, index=False)
