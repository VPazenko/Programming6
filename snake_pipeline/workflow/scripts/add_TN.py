"""
This file creates a random shuffle of the dataset and saves it.
It also creates a true negative set which abides the following rules:
* transgenes have the same amount of cis genes
* cis genes have the same amount of trans genes
* a cis gene can not be a trans gene of itself
"""

__author__ = "Gijs Bakker (modified for snakemake by V.Pazenko)"
__version__ = 1.1

import pandas as pd
# import argparse


def create_random_set(df: pd.DataFrame, index:int):
    """
    This function randomly shuffles the given dataframe of eQTL data
    Args:
        df: a pandas dataframe containing eQTL data
        index: int indicating from where the dataframe needs to bee split and shuffled
    Returns:
        random_shuffle: A pandas dataframe
    """
    first = df.iloc[:,:index]
    # reindex trans_gens so concat merges them in the shuffled order.
    random_shuffle = df.iloc[:,index:].sample(frac=1).reset_index()
    random_shuffle = pd.concat([first, random_shuffle], axis=1)
    return random_shuffle


def create_true_negative_set(df):
    """
    This function will shuffle the database into a true negative set
    Args:
        df: a pandas dataframe containing eQTL data
    Returns:
        true_negatives: A pandas dataframe
    """
    # TODO right now it is possible that by chance the last trans genes remaingin are trans
    # genes of the cis gene or the cis gene itself. To prevent an error, either the function must
    # be re-ran or a smarter way needs to be found
    trans_df = df.trans_eQTL_gene
    true_negatives = pd.DataFrame(columns=['cis_eQTL_gene', 'trans_eQTL_gene'])

    for g in df.cis_eQTL_gene.unique():
        # number of same genes
        amount = sum(df.cis_eQTL_gene == g)
        # get trans genes of the current cis genes
        trans_genes = df.trans_eQTL_gene[df.cis_eQTL_gene == g]
        # add g to prevent genes pointing to themselves.
        trans_genes = pd.concat([trans_genes, pd.Series(g)])
        # grab random trans-genes that are not current genes trans genes
        random_sample = trans_df[~trans_df.isin(trans_genes)].sample(frac=1)[:amount]
        # drop random_sample from trans-genes
        trans_df = trans_df.drop(random_sample.index)
        # concat genes and random sample
        sub_table = random_sample.to_frame().assign(cis_eQTL_gene=g)
        # add to table
        true_negatives = pd.concat([true_negatives, sub_table], axis=0)

    # merge cis data on cis eqtl
    true_negatives = true_negatives.merge(df.iloc[:,:4].drop_duplicates(), how="left", on="cis_eQTL_gene")
    # drop after column 6 since, the probabilities since these probabilities rever to the old matches
    true_negatives = true_negatives.merge(df.iloc[:,4:6].drop_duplicates(), how="left", on="trans_eQTL_gene")
    true_negatives[["posterior_prob", "regional_prob", "candidate_snp", "posterior_explained_by_snp"]] = "NaN"

    # reorder columns
    true_negatives = true_negatives[['cis_eQTL_gene', 'cis_eQTL_gene_name', 'type', 'iteration',
                                    'trans_eQTL_gene', 'trans_eQTL_gene_name', 'posterior_prob',
                                    'regional_prob', 'candidate_snp', 'posterior_explained_by_snp']]
    return true_negatives


def main(eQTL):
    """
    This function will load an eQTL file and save a random shuffle and true negative set.
    """ 

    # random_shuffle = create_random_set(df.iloc[:,:6], 4)
    # random_shuffle.to_csv(f"{output}/RandomShuffle.tsv", index=False, sep="\t")

    negative_set = create_true_negative_set(eQTL)
    negative_set.to_csv(snakemake.output.TN, index=False, sep="\t")

    full_df = pd.concat([eQTL, negative_set])
    full_df.to_csv(snakemake.output.full_df, index=False, sep="\t")


# Load results
eQTL_df = pd.read_csv(snakemake.input[0], sep="\t")
main(eQTL_df)