from cmapPy.pandasGEXpress import parse
import json
import numpy as np

# go through the data set
# for each gene, calculate the variance
# store the top gene and print it out

# load expressions data
def load_landmark_genes():
    lm_genes = json.load(open('landmark_genes.json'))
    ids = []
    for lm_gene in lm_genes:
        ids.append(lm_gene['entrez_id'])
    return ids

def load_gene_expression_data():
    return parse(
        "/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        # col_meta_only=False, row_meta_only=False, rid=['3638'])
        col_meta_only=False, row_meta_only=False, rid=lm_gene_entrez_ids)

lm_gene_entrez_ids = load_landmark_genes()
level_5_gctoo = load_gene_expression_data()

length = len(level_5_gctoo.col_metadata_df.index)

highest_var = 0
highest_gene_id = ""

# For every gene
for gene_id in lm_gene_entrez_ids:
    one_gene_expression_values = []
    for i in range(length):
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        one_gene_expression_values.append(column[gene_id])

    # compute the var
    variance = np.var(one_gene_expression_values)
    print(gene_id)
    print(variance)
    if variance > highest_var:
        highest_var = variance
        highest_gene_id = gene_id

print("Highest:")
print(highest_gene_id)
print(highest_var)