import datetime
import gc
import json
import time
import random
import matplotlib.pyplot as plt
from mlp_optimizer import do_optimize
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv, get_trimmed_feature_dict
from L1000.gene_predictor import train_model, save_model

import helpers.email_notifier as en
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

start_time = time.time()
gene_count_data_limit = 978
use_optimizer = False
target_cell_name = 'VCAP'
direction = 'Both' #'Down'
model_file_prefix = target_cell_name + direction
save_data_to_file = False
use_data_from_file = False

if use_data_from_file:
    prefix = "LDS-1191/saved_xy_data/"
    npX = np.load(prefix + "PC3npX.npz")['arr_0'] # must be not balanced too because 70% of this is X_train.npz
    npY_class = np.load(prefix + "PC3npY_class.npz")['arr_0']
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
drug_features_dict = get_feature_dict('LDS-1191/data/smiles_rdkit_maccs.csv') #, use_int=True)
# drug_descriptor_file = '/data/datasets/gwoo/L1000/LDS-1191/WorkingData/1to12std.csv'
# drug_desc_dict = get_feature_dict(drug_descriptor_file) #, use_int=True)
# print(drug_descriptor_file)
gene_features_dict = get_feature_dict('LDS-1191/data/gene_go_fingerprint_moreThan3.csv')#, use_int=True)
# prot_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/protein_fingerprint.csv')#, use_int=False)
# info to separate by data by cell lines, drug + gene tests may not be equally spread out across cell lines
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
# info to remove any dosages that are not 'µM'. Want to standardize the dosages.
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)

# float16_dict = {}
# for key in prot_features_dict:
#     float16_dict[key] = [float(i) for i in prot_features_dict[key]]
# prot_features_dict = float16_dict

# set maccs keys aside
import os.path
print('remove 1k molecules for validation')
keep_away_filename = 'keep_away.txt'
if os.path.isfile(keep_away_filename):
    keep_away_keys = load_csv(keep_away_filename)
else:
    n_maccs = len(drug_features_dict)
    keep_away_indexes = random.sample(range(0, n_maccs), 1000)
    keys = list(drug_features_dict.keys())
    keep_away_file = open(keep_away_filename, 'w')
    keep_away_keys = []
    for i in keep_away_indexes:
        key = keys[i]
        keep_away_file.write("%s\n" % key)
        keep_away_keys.append([key])
for key in keep_away_keys:
    drug_features_dict.pop(key[0], None)
drug_features_dict.pop("BRD-K56851771", None)

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

# for target_cell_name in ['VCAP', 'HCC515', 'A549', 'HEPG2', 'MCF7', 'HEK293T', 'HT29', 'A375', 'HA1E', 'THP1', 'BT20', 'U937',
#                          'MCF10A', 'HUH7', 'NKDBA', 'NOMO1', 'JURKAT', 'SKBR3', 'HS578T', 'MDAMB231']:
#     for direction in ['Down', 'Up']:
for bin in [10, 5]:
    for target_cell_name in ['VCAP', 'PC3']:#, 'A549', 'MCF7', 'PC3']:
        for direction in ['Multi']:

            model_file_prefix = target_cell_name + '_' + direction + str(bin)
            print(model_file_prefix)

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
                # if drug_id not in drug_desc_dict:
                #     continue
                # more_drug_features = drug_desc_dict[drug_id]

                # parse the dosage unit and value
                dose_unit = experiment_data[5]
                if dose_unit != 'µM': # standardize dose amounts
                    # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
                    continue
                dose_amt = float(experiment_data[4])
                if dose_amt < bin - 0.1 or dose_amt > bin + 0.1: # only use the 5 mm bin
                    continue

                # if dose_amt == 0:
                #     if experiment_data[6] == '1 nM':
                #         dose_amt = 0.001
                #     else:
                #         print("Omitting 0 dose.\n")
                #         continue

                # parse the cell name
                start = find_nth(col_name, "_", 1)
                end = find_nth(col_name, "_", 2)
                cell_name = col_name[start + 1:end]
                if cell_name != target_cell_name:
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

                    pert = column[gene_id].astype('float16')
                    # pert_conc_ratio = abs(pert / dose_amt)
                    abspert = abs(pert)

                    repeat_key = drug_id + "_" + cell_id + "_" + gene_id
                    if repeat_key in repeat_X and abspert <= repeat_X[repeat_key]:
                    # if repeat_key in repeat_X and dose_amt <= repeat_X[repeat_key]:
                        continue

                    if cell_id not in cell_X:
                        cell_X[cell_id] = {}
                        cell_Y[cell_id] = {}
                        cell_drugs_counts[cell_id] = 0
                        cell_Y_gene_ids[cell_id] = []

                    # repeat_X[repeat_key] = pert_conc_ratio
                    repeat_X[repeat_key] = abspert

                    if gene_count_data_limit > 1:
                        cell_X[cell_id][repeat_key] = drug_features + gene_features_dict[gene_symbol]# + more_drug_features + prot_features_dict[gene_symbol]
                    else:
                        cell_X[cell_id][repeat_key] = drug_features
                    cell_Y[cell_id][repeat_key] = pert
                    cell_Y_gene_ids[cell_id].append(gene_id)
                    cell_drugs_counts[cell_id] += 1

            elapsed_time = time.time() - start_time
            print(datetime.datetime.now(), "Time to load data:", elapsed_time)

            gene_cutoffs_down = {}
            gene_cutoffs_up = {}
            percentile_down = 5 # for downregulation, use 95 for upregulation
            percentile_up = 95

            # use_global gene_specific_cutoffs:
            prog_ctr = 0
            for gene_id in lm_gene_entrez_ids:
                row = level_5_gctoo.data_df.loc[gene_id, :].values
                prog_ctr += 1
                printProgressBar(prog_ctr, gene_count_data_limit, prefix='Storing percentile cutoffs')
                gene_cutoffs_down[gene_id] = np.percentile(row, percentile_down)
                gene_cutoffs_up[gene_id] = np.percentile(row, percentile_up)

            gc.collect()
            cell_line_counter = 1
            print(datetime.datetime.now(), "Gene count:", gene_count_data_limit, "\n")
            try:
                for cell_name in cell_name_to_id_dict:
                    cell_id = cell_name_to_id_dict[cell_name][0]
                    if cell_id not in cell_X:
                        continue
                    print(datetime.datetime.now(), "Converting dictionary values to np")
                    npX = np.asarray(list(cell_X[cell_id].values()), dtype='float16')
                    npY = np.asarray(list(cell_Y[cell_id].values()), dtype='float16')
                    npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])

                    npY_class = np.zeros(len(npY), dtype=int)

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
                        if direction == 'Down':
                            intersect = np.intersect1d(gene_locations, down_locations)
                            npY_class[intersect] = 1
                        elif direction == 'Up':
                            intersect = np.intersect1d(gene_locations, up_locations)
                            npY_class[intersect] = 1
                        elif direction == 'Both':
                            intersect = np.intersect1d(gene_locations, down_locations)
                            combined_locations += intersect.tolist()
                            intersect = np.intersect1d(gene_locations, up_locations)
                            combined_locations += intersect.tolist()
                            npY_class[intersect] = 1
                        else: # direction = multi
                            intersect = np.intersect1d(gene_locations, down_locations)
                            gene_down_locations = intersect.tolist()
                            combined_locations += gene_down_locations
                            intersect = np.intersect1d(gene_locations, up_locations)
                            gene_up_locations = intersect.tolist()
                            combined_locations += gene_up_locations
                            npY_class[gene_up_locations] = 2
                            npY_class[gene_down_locations] = 1

                    if direction == 'Both':
                        npX = npX[combined_locations]
                        npY_class = npY_class[combined_locations]
                    print("Evaluating cell line", cell_line_counter, cell_name, "(Percentile ends:", percentile_down, ")")

                    sample_size = len(npY_class)

                    if sample_size < 300: # smaller sizes was giving y values of only one class
                        continue

                    print('Positive samples', np.sum(npY_class))

                    num_drugs = cell_drugs_counts[cell_id]
                    print("Sample Size:", sample_size, "Drugs tested:", num_drugs / gene_count_data_limit)

                    if save_data_to_file:
                        prefix = "LDS-1191/saved_xy_data/"
                        np.savez(prefix + cell_name + "npXEndsAllCutoffs", npX)
                        np.savez(prefix + cell_name + "npY_classEndsAllCutoffs", npY_class)

                    if use_optimizer:
                        do_optimize(len(np.unique(npY_class)), npX, npY_class)
                    else:
                        model = train_model(npX, npY_class)
                        save_model(model, model_file_prefix)
            finally:
                en.notify()
                plt.show()