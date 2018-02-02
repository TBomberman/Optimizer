from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar
import json
from random_forest import do_optimize
import numpy as np
import gc

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('one_landmark_gene.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

# get the dictionaries
# get the expressions
print("Loading drug and gene features")
drug_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/smiles_rdkit_maccs.csv', use_int=True)
cell_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/cell_line_fingerprint.csv', use_int=True)
gene_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/gene_go_fingerprint.csv', use_int=True)
batch_to_drug_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Small_Molecule_Metadata.txt', '\t', 9)
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 0)
gene_id_dict = get_gene_id_dict()
lm_gene_entrez_ids = list(gene_id_dict.keys())[:50]
print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)

cell_X = {}
cell_Y = {}
cell_drugs = {}

# For every experiment
print("Loading experiments")
one_percent = int(length/50)
for i in range(length):
    if i % one_percent == 0:
        printProgressBar(i, length, prefix='Progress:', suffix='Complete', length=50)
    X = []
    Y = []

    col_name = level_5_gctoo.col_metadata_df.index[i]
    column = level_5_gctoo.data_df[col_name]

    # parse the batch id
    start = find_nth(col_name, ":", 1)
    end = find_nth(col_name, ":", 2)
    batch_id = col_name[start + 1:end]
    if batch_id not in batch_to_drug_id_dict:
        continue
    drug_id = batch_to_drug_id_dict[batch_id][7]
    if drug_id not in drug_features_dict:
        continue
    drug_features = drug_features_dict[drug_id]

    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]
    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]
    if cell_id not in cell_features_dict:
        continue
    cell_features = cell_features_dict[cell_id]

    for gene_id in lm_gene_entrez_ids:
        gene_symbol = gene_id_dict[gene_id]

        if gene_symbol not in gene_features_dict:
            continue

        if cell_id not in cell_X:
            cell_X[cell_id] = []
            cell_Y[cell_id] = []
            cell_drugs[cell_id] = []

        cell_X[cell_id].append(drug_features + gene_features_dict[gene_symbol])
        cell_Y[cell_id].append(column[gene_id])
        cell_drugs[cell_id].append(drug_id)

gc.collect()
count = 1
for cell_name in cell_name_to_id_dict:
    cell_id = cell_name_to_id_dict[cell_name][0]
    if cell_id not in cell_X:
        print("Skipping", cell_name, ". No cell line data.\n")
        continue

    npX = np.asarray(cell_X[cell_id])
    npY = np.asarray(cell_Y[cell_id])

    sample_size = len(npX)

    if sample_size < 3:
        print("Skipping", cell_name, ". Sample size", sample_size, "is too small.\n")
        continue

    npY_class = np.zeros(len(npY), dtype=int)
    npY_class[np.where(npY > 0)] = 1

    # np.savetxt("x.csv", npX, fmt='%i')
    # np.savetxt("y.csv", npY_class, fmt='%i')

    print("Evaluating cell line", count, cell_name)
    print("Sample Size:", sample_size, "Drugs tested:", len(set(cell_drugs[cell_id])))
    do_optimize(0, npX, npY_class)
    count += 1
    print()
