import datetime
import json
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv

gene_count_data_limit = 978

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

print(datetime.datetime.now(), "Loading drug and gene features")
drug_features_dict = get_feature_dict('data/smiles_rdkit_maccs.csv') #, use_int=True)
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
keep_away_filename = '../keep_away.txt'
keep_away_keys = load_csv(keep_away_filename)
keep_away_keys = np.asarray(keep_away_keys).reshape((1000))

# getting the gene ids
gene_id_dict = get_gene_id_dict()
# lm_gene_entrez_ids = list(gene_id_dict.keys())[:200]
lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)


print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)
repeat_X = {}
perts = {}
pert_counts = 0

# For every experiment
print("Loading experiments")
for i in range(length-1, -1, -1): # go backwards, assuming later experiments have stronger perturbation
    printProgressBar(length - i, length, prefix='Load experiments progress')
    col_name = level_5_gctoo.col_metadata_df.index[i]
    column = level_5_gctoo.data_df[col_name]

    # parse the time
    start = col_name.rfind("_")
    end = find_nth(col_name, ":", 1)
    exposure_time = col_name[start + 1:end]
    if exposure_time != "24H": # column counts: 6h 95219, 24h 109287, 48h 58, 144h 1
        continue

    # get drug features
    col_name_key = col_name[2:-1]
    if col_name_key not in experiments_dose_dict:
        continue
    experiment_data = experiments_dose_dict[col_name_key]
    drug_id = experiment_data[0]
    if drug_id not in drug_features_dict:
        continue

    if drug_id not in keep_away_keys:
        continue

    # parse the dosage unit and value
    dose_unit = experiment_data[5]
    if dose_unit != 'µM':
        continue
    dose_amt = float(experiment_data[4])
    if dose_amt < 4.9 or dose_amt > 5.1:  # only use the 5 mm bin
        continue

    # parse the cell name
    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]
    if cell_name != 'PC3':
        continue
    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]

    for gene_id in lm_gene_entrez_ids:
        pert = column[gene_id]
        abspert = abs(pert)

        repeat_key = drug_id + "_" + cell_id
        if repeat_key in repeat_X and abspert <= repeat_X[repeat_key]:
            continue
        repeat_X[repeat_key] = abspert

        if drug_id not in perts:
            perts[drug_id] = {}
        if gene_id not in perts[drug_id]:
            perts[drug_id][gene_id] = pert
            pert_counts += 1

print('pert counts', str(pert_counts))
corr_x = []
corr_y = []

from pathlib import Path
from L1000.gene_predictor import load_model

up_models = []
down_models = []
path_prefix = "saved_models/"
gene_features_dict = get_feature_dict('data/gene_go_fingerprint_moreThan3.csv', use_int=True)

down_model_file_prefixes = [
    'PC3Down5bin-1k77'
]

up_model_file_prefixes = [
    'PC3Up5bin-1k78'
]

# load model
def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)

up_model = load_model_from_file_prefix(path_prefix + up_model_file_prefixes[i])
down_model = load_model_from_file_prefix(path_prefix + down_model_file_prefixes[i])

all_samples = []
for drug_id in perts.keys():
    if drug_id not in drug_features_dict:
        continue
    drug_features = drug_features_dict[drug_id]
    for gene_id in perts[drug_id].keys():
        gene_symbol = gene_id_dict[gene_id]
        if gene_symbol not in gene_features_dict:
            continue
        gene_features = gene_features_dict[gene_symbol]
        all_samples.append(np.asarray(drug_features + gene_features))
        corr_y.append(perts[drug_id][gene_id])
np_all_samples = np.asarray(all_samples)
up_model_all_predictions = up_model.predict(np_all_samples)
down_model_all_predictions = down_model.predict(np_all_samples)
flipped_down_predictions = np.fliplr(down_model_all_predictions)
all_predictions = np.mean(np.array([up_model_all_predictions, flipped_down_predictions]), axis=0)
for prediction in all_predictions:
    corr_x.append(prediction[1])

print('corr y length', len(corr_y))
print('corr x length', len(corr_x))

# x = []
# for val in corr_x:
#     x.append((val - 0.5)*10)

x = corr_x
y = corr_y

import matplotlib.pyplot as plt
from helpers.utilities import scatter2D_plot, scatterdens_plot
scatter2D_plot(y, x, file="Corr")
plt.show()
