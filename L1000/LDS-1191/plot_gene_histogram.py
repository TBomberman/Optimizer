import json

import matplotlib.pyplot as plt
from L1000.data_loader import load_gene_expression_data

# go through the data set
# for each gene, calculate the variance
# store the top gene and print it out

# load expressions data
def load_landmark_genes():
    symbols = {}
    lm_genes = json.load(open('data/landmark_genes.json'))
    for lm_gene in lm_genes:
        symbols[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return symbols

symbols = load_landmark_genes()
lm_gene_entrez_ids = ['2778']
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)

# For every gene
for gene_id in lm_gene_entrez_ids:
    one_gene_expression_values = []
    for i in range(length):
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        one_gene_expression_values.append(column[gene_id])

    plt.title('Histogram of gene: GNAS')
    plt.ylabel('Number of expressions')
    plt.xlabel('Z-score')
    plt.hist(one_gene_expression_values, bins=1000)
    plt.show()
