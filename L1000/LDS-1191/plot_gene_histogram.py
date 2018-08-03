import json
import matplotlib.pyplot as plt
import numpy as np
from L1000.data_loader import load_gene_expression_data, load_csv, printProgressBar

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
gene_count_data_limit = 978
lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)
# lm_gene_entrez_ids = ['2778']
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)
# length = 10
one_gene_expression_values = []

# For every gene
for i in range(length):
    printProgressBar(i, length, prefix='Load experiments progress')
    col_name = level_5_gctoo.col_metadata_df.index[i]
    column = level_5_gctoo.data_df[col_name]
    for gene_id in lm_gene_entrez_ids:
        one_gene_expression_values.append(column[gene_id])

min = np.min(one_gene_expression_values)
max = np.max(one_gene_expression_values)
mean = np.mean(one_gene_expression_values)
median = np.median(one_gene_expression_values)
print('min', min)
print('max', max)
print('mean', mean)
print('median', median)

q5 = np.percentile(one_gene_expression_values, 5)
q25 = np.percentile(one_gene_expression_values, 25)
q50 = np.percentile(one_gene_expression_values, 50)
q75 = np.percentile(one_gene_expression_values, 75)
q95 = np.percentile(one_gene_expression_values, 95)

print("5% quantile: {}\n25% quantile: {}\n50% quantile: {}\n75% quantile: {}\n95% quantile: {}\n".format(q5, q25, q50, q75, q95))

plt.title('Histogram of Perturbations')
plt.ylabel('Number of expressions')
plt.xlabel('Z-score')
plt.hist(one_gene_expression_values, bins=1000)
plt.show()
