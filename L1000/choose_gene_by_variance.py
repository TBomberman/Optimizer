from L1000.data_loader import load_gene_expression_data
import json
import numpy as np
from operator import itemgetter
import csv
import matplotlib.pyplot as plt

# go through the data set
# for each gene, calculate the variance
# store the top gene and print it out

plot_historgram = False

# load expressions data
def load_landmark_genes():
    lm_genes = json.load(open('landmark_genes.json'))
    ids = []
    for lm_gene in lm_genes:
        ids.append(lm_gene['entrez_id'])
    return ids

lm_gene_entrez_ids = load_landmark_genes()
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

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

with open("genes_by_var.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in gene_ids:
        writer.writerow([val])
