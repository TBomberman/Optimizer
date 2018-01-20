from L1000.data_loader import get_feature_dict, load_gene_expression_data
import json
from random_forest import do_optimize
import numpy as np

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

# get the dictionaries
# get the expressions
drug_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/smiles_rdkit_maccs.csv')
cell_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/cell_line_fingerprint.csv')
gene_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/gene_go_fingerprint.csv')
batch_to_drug_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Small_Molecule_Metadata.txt', '\t', 9)
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 0)
gene_id_dict = get_gene_id_dict()
lm_gene_entrez_ids = list(gene_id_dict.keys())
level_5_gctoo = load_gene_expression_data(lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)
X = []
Y = []

# For every experiment
for i in range(length):
    col_name = level_5_gctoo.col_metadata_df.index[i]
    column = level_5_gctoo.data_df[col_name]

    # parse the batch id
    start = find_nth(col_name, ":", 1)
    end = find_nth(col_name, ":", 2)
    batch_id = col_name[start + 1:end]
    if batch_id not in batch_to_drug_id_dict:
        continue
    drug_id = batch_to_drug_id_dict[batch_id][7]

    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]
    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]

    for gene_id in lm_gene_entrez_ids:
        gene_symbol = gene_id_dict[gene_id]

        if drug_id not in drug_features_dict:
            continue
        if cell_id not in cell_features_dict:
            continue
        if gene_symbol not in gene_features_dict:
            continue
        X.append(drug_features_dict[drug_id] + cell_features_dict[cell_id] + gene_features_dict[gene_symbol])
        Y.append(column[gene_id])

npX = np.asarray(X)
npY = np.asarray(Y)
# np.savetxt("x.csv", X)
# np.savetxt("y.csv", Y)
do_optimize(0, npX, npY)
