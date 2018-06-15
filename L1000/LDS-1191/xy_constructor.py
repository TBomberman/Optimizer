import datetime
import gc
import json
import time

import matplotlib.pyplot as plt
from mlp_optimizer import do_optimize
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv, get_trimmed_feature_dict
from L1000.gene_predictor import train_model, save_model

import helpers.email_notifier as en

start_time = time.time()
gene_count_data_limit = 977
use_optimizer = True
model_file_prefix = "100PC3PD"
save_data_to_file = False
use_data_from_file = False
use_gene_specific_cutoffs = True

if use_data_from_file:
    npX = np.load("PC3npX.npz")['arr_0'] # must be not balanced too because 70% of this is X_train.npz
    npY_class = np.load("PC3npY_class.npz")['arr_0']
    try:
        if use_optimizer:
            do_optimize(2, npX, npY_class)
        else:
            model = train_model(npX, npY_class)
            save_model(model, model_file_prefix)
    finally:
        en.notify()
        plt.show()
        quit()

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('LDS-1191/data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

# get the dictionaries
# get the expressions
print(datetime.datetime.now(), "Loading drug and gene features")
drug_features_dict = get_feature_dict('LDS-1191/data/smiles_rdkit_maccs_trimmed.csv') #, use_int=True)
gene_features_dict = get_feature_dict('LDS-1191/data/gene_go_fingerprint.csv')#, use_int=True)
# prot_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/protein_fingerprint.csv')#, use_int=False)
# info to separate by data by cell lines, drug + gene tests may not be equally spread out across cell lines
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
# info to remove any dosages that are not 'µM'. Want to standardize the dosages.
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
cell_features_dict = get_feature_dict('LDS-1191/data/cell_line_fingerprint.csv')

# getting the gene ids
gene_id_dict = get_gene_id_dict()
# lm_gene_entrez_ids = list(gene_id_dict.keys())[:200]
lm_gene_entrez_ids_list = load_csv('LDS-1191/data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)


print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)

cell_X = {}
cell_Y = {}
cell_Y_gene_ids = {}
cell_drugs_counts = {}
repeat_X = {}

# For every experiment
print("Loading experiments")
for i in range(length-1, -1, -1): # go backwards, assuming later experiments have stronger perturbation
    printProgressBar(length - i, length, prefix='Load experiments progress')
    X = []
    Y = []

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
    drug_features = drug_features_dict[drug_id]

    # parse the dosage unit and value
    dose_unit = experiment_data[5]
    if dose_unit != 'µM': # standardize dose amounts
        # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
        continue
    dose_amt = float(experiment_data[4])
    if dose_amt == 0:
        if experiment_data[6] == '1 nM':
            dose_amt = 0.001
        else:
            print("Omitting 0 dose.\n")
            continue

    # parse the cell name
    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]
    # cell_name = "A375"  # this line will combine all cell lines into one
    if cell_name != 'PC3':
        continue

    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]

    for gene_id in lm_gene_entrez_ids:
        gene_symbol = gene_id_dict[gene_id]

        if gene_symbol not in gene_features_dict:
            continue

        # if gene_symbol not in prot_features_dict:
        #     continue

        if cell_id not in cell_features_dict:
            continue

        if cell_id not in cell_X:
            cell_X[cell_id] = {}
            cell_Y[cell_id] = {}
            cell_drugs_counts[cell_id] = 0
            cell_Y_gene_ids[cell_id] = []

        repeat_key = drug_id + "_" + cell_id + "_" + gene_id
        if repeat_key in repeat_X and dose_amt <= repeat_X[repeat_key]:
            # print("repeat_key", repeat_key, "dose amount", dose_amt, "is less than", repeat_X[repeat_key])
            continue
        repeat_X[repeat_key] = dose_amt

        if gene_count_data_limit > 1:
            # cell_X[cell_id].append([dose_amt] + drug_features + cell_features_dict[cell_id] + gene_features_dict[gene_symbol])
            cell_X[cell_id][repeat_key] = drug_features + gene_features_dict[gene_symbol]#+ prot_features_dict[gene_symbol]
        else:
            cell_X[cell_id][repeat_key] = drug_features
        pert = column[gene_id].astype('float16')
        cell_Y[cell_id][repeat_key] = pert
        cell_Y_gene_ids[cell_id].append(gene_id)
        cell_drugs_counts[cell_id] += 1

elapsed_time = time.time() - start_time
print("Time to load data:", elapsed_time)

gene_cutoffs_down = {}
gene_cutoffs_up = {}
percentile_down = 5 # for downregulation, use 95 for upregulation
percentile_up = 95

prog_ctr = 0
for gene_id in lm_gene_entrez_ids:
    row = level_5_gctoo.data_df.loc[gene_id, :].values
    prog_ctr += 1
    printProgressBar(prog_ctr, gene_count_data_limit, prefix='Storing percentile cutoffs')
    gene_cutoffs_down[gene_id] = np.percentile(row, percentile_down)
    gene_cutoffs_up[gene_id] = np.percentile(row, percentile_up)

del level_5_gctoo
gc.collect()
cell_line_counter = 1
print("Gene count:", gene_count_data_limit, "\n")
try:
    for cell_name in cell_name_to_id_dict:
        # print("Looking at", cell_name)
        cell_id = cell_name_to_id_dict[cell_name][0]
        if cell_id not in cell_X:
            # print("Skipping", cell_name, ". No cell line data.\n")
            continue
        npX = []
        npY = []
        listKeys = []
        for key, value in cell_X[cell_id].items():
            npX.append(value)
            # listKeys.append(key.split("_")[0])
        for key, value in cell_Y[cell_id].items():
            npY.append(value)

        npX = np.asarray(npX, dtype='float16')
        npY = np.asarray(npY, dtype='float16')
        npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])

        npY_class = np.zeros(len(npY), dtype=int)
        if use_gene_specific_cutoffs:
            prog_ctr = 0
            combined_locations = []
            for gene_id in lm_gene_entrez_ids: # this section is for gene specific class cutoffs
                prog_ctr += 1
                printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
                class_cut_off_down = gene_cutoffs_down[gene_id]
                class_cut_off_up = gene_cutoffs_up[gene_id]
                gene_locations = np.where(npY_gene_ids == gene_id)
                down_locations = np.where(npY <= class_cut_off_down)
                up_locations = np.where(npY >= class_cut_off_up)
                intersect = np.intersect1d(gene_locations, down_locations)
                combined_locations += intersect.tolist()
                intersect = np.intersect1d(gene_locations, up_locations)
                npY_class[intersect] = 1
                combined_locations += intersect.tolist()
            npX = npX[combined_locations]
            npY_class = npY_class[combined_locations]
            print("size after top down split", len(npY_class))
            print("Evaluating cell line", cell_line_counter, cell_name, "(Percentile ends:", percentile_down, ")")
        else:
            npY_class[np.where(npY > class_cut_off_down)] = 1 # generic class cutoff
            print("Evaluating cell line", cell_line_counter, cell_name, "class cutoff", class_cut_off_down, "(Percentile ends:", percentile_down, ")")

        sample_size = len(npY_class)

        if sample_size < 300: # smaller sizes was giving y values of only one class
            # print("Skipping", cell_name, ". Sample size", sample_size, "is too small.\n")
            continue

        print(np.sum(npY_class))

        num_drugs = cell_drugs_counts[cell_id]
        print("Sample Size:", sample_size, "Drugs tested:", num_drugs)

        # count number of perturbed genes
        # key_set = set(listKeys)
        # set_keys = np.array(listKeys)
        # num_gene_effects = []
        # for drug_key in key_set:
        #     drug_idx = np.where(set_keys == drug_key)[0].tolist()
        #     num_gene_effects.append(gene_count_data_limit - np.count_nonzero(npY_class[drug_idx]))
        # plt.hist(num_gene_effects, bins='auto')
        # plt.draw()
        # print(drug_key, num_gene_effects)

        if save_data_to_file:
            np.savez(cell_name + "npX", npX)
            np.savez(cell_name + "npY_class", npY_class)

        if use_optimizer:
            do_optimize(2, npX, npY_class) #, listKeys)
        else:
            model = train_model(npX, npY_class)
            save_model(model, model_file_prefix)
finally:
    en.notify()
    plt.show()