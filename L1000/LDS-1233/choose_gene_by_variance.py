import csv
import json
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from L1000.data_loader import load_gene_expression_data

# go through the data set
# for each gene, calculate the variance
# store the top gene and print it out

plot_historgram = False

# load expressions data
def load_landmark_genes():
    lm_genes = json.load(open('data/landmark_genes.json'))
    ids = []
    for lm_gene in lm_genes:
        ids.append(get_their_id(lm_gene['entrez_id']))
    return ids

def get_their_id(good_id):
    return 'b\'' + good_id + '\''

def get_our_id(bad_id):
    return bad_id[2:-1]

lm_gene_entrez_ids = load_landmark_genes()
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1233/Data/GSE70138_Broad_LINCS_Level4_ZSVCINF_mlr12k_n78980x22268_2015-06-30.gct")#, lm_gene_entrez_ids)
row_headers = level_5_gctoo.col_metadata_df.values


length = len(level_5_gctoo.col_metadata_df.index)
highest_var = 0
highest_gene_id = ""
gene_var = []

# For every gene
for gene_id in lm_gene_entrez_ids:
    one_gene_expression_values = []
    for i in range(length):
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        one_gene_expression_values.append(column[gene_id])

    # plot a histogram for this gene
    if plot_historgram:
        plt.title('Histogram of geneid ' + gene_id)
        plt.ylabel('Number of expressions')
        plt.xlabel('Z-score')
        plt.hist(one_gene_expression_values, bins=1000)
        plt.show()

    # compute the var
    variance = np.var(one_gene_expression_values)
    gene_var.append([gene_id, variance])
    print(gene_id)
    print(variance)
    if variance > highest_var:
        highest_var = variance
        highest_gene_id = gene_id

print("Highest:")
print(highest_gene_id)
print(highest_var)

gene_var.sort(key=itemgetter(1), reverse=True)
gene_ids = [tup[0] for tup in gene_var]

with open("data/genes_by_var.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in gene_ids:
        writer.writerow([val])
