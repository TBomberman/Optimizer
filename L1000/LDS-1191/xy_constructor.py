import datetime
import gc
import json
import time
import random
import matplotlib.pyplot as plt
from ensemble_optimizer import do_optimize
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv, get_trimmed_feature_dict
from L1000.gene_predictor import train_model, save_model
from sklearn.model_selection import train_test_split
import helpers.email_notifier as en
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

start_time = time.time()
gene_count_data_limit = 978
evaluate_type = "use_optimizer" #"use_optimizer" "train_and_save" "test_trained"
# target_cell_names = ['PC3', 'HT29']
# target_cell_names = ['MCF7', 'A375']
# target_cell_names = ['VCAP', 'A549']
target_cell_names = ['A375']
direction = 'Multi' #'Down'
save_data_to_file = False
use_data_from_file = True
test_blind = False
load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/morgan2048/5p/"
data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/cv/morgan2048/5p/"
# gap_factors = [0.8, 0.6, 0.4, 0.2, 0.0]
# gap_factors = [0.2, 0.3] #, 0.6, 0.4, 0.2, 0.0]
# gap_factors = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9]
gap_factors = [0.0]
# class_weights = [0.01, 0.03, 0.05, 0.1, 0.15]
percentiles = [5]
class_weights = [0.01]
if use_data_from_file:
    for target_cell_name in target_cell_names:
        for bin in [10]:
            for percentile_down in percentiles:
                for gap_factor in gap_factors:
                    for class_0_weight in class_weights:
                        file_suffix = target_cell_name + '_' + direction + '_' + str(bin) + 'b_' + str(percentile_down) + \
                                      'p_' + str(int(gap_factor*100)) + 'g'

                        model_file_prefix = data_folder_path + str(datetime.datetime.now()) + '_' + file_suffix + \
                                            '_' + str(int(class_0_weight*100)) + 'c'
                        print('load location', load_data_folder_path)
                        print('save location', model_file_prefix)
                        npX = np.load(load_data_folder_path + file_suffix + "_npX.npz")['arr_0'] # must be not balanced too because 70% of this is X_train.npz
                        npY_class = np.load(load_data_folder_path + file_suffix + "_npY_class.npz")['arr_0']
                        # cold_ids = np.load(load_data_folder_path + file_suffix + "_cold_ids.npz")['arr_0']
                        # npY = np.load(load_data_folder_path + file_suffix + "_npY_float.npz")['arr_0']
                        # test_npX = np.load(load_data_folder_path + file_suffix + "_test_npX.npz")['arr_0']
                        # test_npY_class = np.load(load_data_folder_path + file_suffix + "_test_npY_class.npz")['arr_0']
                        # test_npY_float = np.load(load_data_folder_path + file_suffix + "_test_npY_float.npz")['arr_0']
                        npY = []
                        test_npX = []
                        test_npY_class = []
                        test_npY_float = []
                        cold_ids = []

                        # for testing
                        # ints = random.sample(range(1,100000), 1000)
                        # npX = npX[ints]
                        # npY_class = npY_class[ints]
                        # cold_ids = cold_ids[ints]

                        try:
                            if evaluate_type == "use_optimizer":
                                do_optimize(len(np.unique(npY_class)), npX, npY_class, model_file_prefix, class_0_weight,
                                            cold_ids, labels_float=npY) #, test_data=[test_npX, test_npY_class, test_npY_float])
                            elif evaluate_type == "train_and_save":
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
drug_features_dict = get_feature_dict('LDS-1191/data/smiles_rdkit_morgan_2048.csv') #, use_int=True)
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

# set 1000 blind maccs keys aside
import os.path
print('remove blind drugs for validation')
blind_drugs_filename = 'LDS-1191/data/blind_drugs.csv'
if os.path.isfile(blind_drugs_filename):
    blind_drugs_keys = load_csv(blind_drugs_filename)
else:
    n_maccs = len(drug_features_dict)
    blind_drugs_indexes = random.sample(range(0, n_maccs), 1000)
    keys = list(drug_features_dict.keys())
    blind_drugs_file = open(blind_drugs_filename, 'w')
    blind_drugs_keys = []
    for i in blind_drugs_indexes:
        key = keys[i]
        blind_drugs_file.write("%s\n" % key)
        blind_drugs_keys.append([key])

# if test_blind:
#     keys_to_remove = []
#     for key in drug_features_dict.keys():
#         if [key] in blind_drugs_keys:
#             continue
#         keys_to_remove.append(key)
#     for key in keys_to_remove:
#         drug_features_dict.pop(key, None)
# else:
#     for key in blind_drugs_keys:
#         drug_features_dict.pop(key[0], None)

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
# length = 10000

# for target_cell_name in ['VCAP', 'HCC515', 'A549', 'HEPG2', 'MCF7', 'HEK293T', 'HT29', 'A375', 'HA1E', 'THP1', 'BT20', 'U937',
#                          'MCF10A', 'HUH7', 'NKDBA', 'NOMO1', 'JURKAT', 'SKBR3', 'HS578T', 'MDAMB231']:
#     for direction in ['Down', 'Up']:

for target_cell_name in target_cell_names:
    for bin in [10]:
        for direction in ['Multi']: # 'Multi' 'Both' 'Up' 'Down'
            cell_X = {}
            cell_cold_ids = {} # basically just storing the ids of each sample so you can know which drug or gene it
            # refers to
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
                        cell_cold_ids[cell_id] = {}
                        cell_Y[cell_id] = {}
                        cell_drugs_counts[cell_id] = 0
                        cell_Y_gene_ids[cell_id] = []

                    updated = False
                    # repeat_X[repeat_key] = pert_conc_ratio
                    if repeat_key in repeat_X:
                        updated = True
                    repeat_X[repeat_key] = abspert

                    if gene_count_data_limit > 1:
                        cell_X[cell_id][repeat_key] = drug_features + gene_features_dict[gene_symbol]# + more_drug_features + prot_features_dict[gene_symbol]
                        cell_cold_ids[cell_id][repeat_key] = drug_id
                    else:
                        cell_X[cell_id][repeat_key] = drug_features
                        cell_cold_ids[cell_id][repeat_key] = drug_id
                    if repeat_key not in cell_Y[cell_id]:
                        cell_Y[cell_id][repeat_key] = []
                    cell_Y[cell_id][repeat_key].append(pert)
                    if not updated:
                        cell_Y_gene_ids[cell_id].append(gene_id)
                        cell_drugs_counts[cell_id] += 1

            elapsed_time = time.time() - start_time
            print(datetime.datetime.now(), "Time to load data:", elapsed_time)

            gene_cutoffs_down = {}
            gene_cutoffs_up = {}
            # percentile_down = 5 # for downregulation, use 95 for upregulation
            for percentile_down in percentiles:
                percentile_up = 100 - percentile_down

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
                        cold_ids = list(cell_cold_ids[cell_id].values())
                        y_pert_lists = np.asarray(list(cell_Y[cell_id].values()))
                        # npY = np.asarray(list(cell_Y[cell_id].values()), dtype='float16')
                        npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])
                        n_samples = len(y_pert_lists)
                        npY_class = np.zeros(n_samples, dtype=int)

                        for gap_factor in gap_factors:
                            model_file_prefix = load_data_folder_path + target_cell_name + '_' + direction + '_' + \
                                                str(bin) + 'b_' + str(percentile_down) + 'p_' + \
                                                str(int(gap_factor * 100)) + 'g'
                            print(model_file_prefix)
                            prog_ctr = 0
                            combined_locations = []
                            combined_test_locations = []
                            for gene_id in lm_gene_entrez_ids: # this section is for gene specific class cutoffs
                                prog_ctr += 1
                                printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
                                class_cut_off_down = gene_cutoffs_down[gene_id]
                                class_cut_off_up = gene_cutoffs_up[gene_id]
                                gene_locations = np.where(npY_gene_ids == gene_id)
                                # dummy, gene_test_locations = train_test_split(gene_locations[0], train_size=0.85,
                                #                                                        test_size=0.15, shuffle=False)
                                down_threshold = class_cut_off_down # - abs(gap_factor * class_cut_off_down)
                                up_threshold = class_cut_off_up # + abs(gap_factor * class_cut_off_up)
                                mid_threshold_bottom = class_cut_off_down + abs(gap_factor * class_cut_off_down)
                                mid_threshold_top = class_cut_off_up - abs(gap_factor * class_cut_off_up)
                                # print("gene", gene_id, "down", down_threshold, "mid bottom", mid_threshold_bottom, "mid top", mid_threshold_top, "up", up_threshold)

                                def get_class_vote(pert_list, bottom_threshold, top_threshold,
                                                mid_bottom_threshold, mid_top_threshold):
                                    votes = [0,0,0,0]
                                    # list of perts
                                    for pert in pert_list:
                                        if pert > top_threshold:
                                            votes[2] += 1
                                        elif pert < bottom_threshold:
                                            votes[1] += 1
                                        elif mid_bottom_threshold < pert < mid_top_threshold:
                                            votes[3] += 1
                                        else:
                                            votes[0] += 1
                                    is_tie = False
                                    highest_vote_class = np.argmax(votes)
                                    for i in range(0, len(votes)):
                                        if i == highest_vote_class:
                                            continue
                                        if votes[i] == votes[highest_vote_class]:
                                            is_tie = True
                                            break
                                    if is_tie:
                                        return 0
                                    else:
                                        return highest_vote_class
                                class_votes = np.zeros(n_samples, dtype=int)

                                for pert_i in gene_locations[0]:
                                    class_votes[pert_i] = get_class_vote(y_pert_lists[pert_i], down_threshold, up_threshold,
                                                              mid_threshold_bottom, mid_threshold_top)
                                down_locations = np.where(class_votes == 1)
                                up_locations = np.where(class_votes == 2)
                                mid_locations = np.where(class_votes == 3)
                                if direction == 'Down':
                                    npY_class[down_locations] = 1
                                elif direction == 'Up':
                                    npY_class[up_locations] = 1
                                elif direction == 'Both':
                                    combined_locations += down_locations[0].tolist()
                                    combined_locations += up_locations.tolist()
                                    npY_class[combined_locations] = 1
                                else: # direction = multi
                                    combined_locations += down_locations[0].tolist()
                                    combined_locations += up_locations[0].tolist()
                                    combined_locations += mid_locations[0].tolist()
                                    npY_class[up_locations] = 2
                                    npY_class[down_locations] = 1
                                    # combined_test_locations += gene_test_locations.tolist()
                                    # print(len(gene_down_locations), "samples below", down_threshold)
                                    # print(len(mid_locations), "samples between", mid_threshold_bottom, "and", mid_threshold_top)
                                    # print(len(gene_up_locations), "samples below", up_threshold)
                                    # print("down samples", len(gene_down_locations), "up samples", len(gene_up_locations), "mid samples", len(mid_locations))
                            # print("total samples", len(combined_locations))

                            if direction == 'Both' or direction == 'Multi':
                                npX_save = npX[combined_locations]
                                cold_ids_save = [cold_ids[ci] for ci in combined_locations]
                                npY_class_save = npY_class[combined_locations]
                                # npY_save = npY_class[combined_locations]
                                # test_npX_save = npX[combined_test_locations]
                                # test_npY_class_save = npY_class[combined_test_locations]
                                # test_npY_save = npY_class[combined_test_locations]
                            print("Evaluating cell line", cell_line_counter, cell_name, "(Percentile ends:", percentile_down, ")")

                            sample_size = len(npY_class_save)

                            if sample_size < 300: # smaller sizes was giving y values of only one class
                                continue

                            print('Positive samples', np.sum(npY_class_save))

                            num_drugs = cell_drugs_counts[cell_id]
                            print("Sample Size:", sample_size, "Drugs tested:", num_drugs / gene_count_data_limit)

                            if save_data_to_file:
                                np.savez(model_file_prefix + "_npX", npX_save)
                                np.savez(model_file_prefix + "_npY_class", npY_class_save)
                                np.savez(model_file_prefix + "_cold_ids", cold_ids_save)
                                # np.savez(model_file_prefix + "_npY_float", npY_save)
                                # np.savez(model_file_prefix + "_test_npX", test_npX_save)
                                # np.savez(model_file_prefix + "_test_npY_class", test_npY_class_save)
                                # np.savez(model_file_prefix + "_test_npY_float", test_npY_save)

                            # if evaluate_type == "use_optimizer":
                            #     do_optimize(len(np.unique(npY_class)), npX, npY_class, model_file_prefix, None, cold_ids, labels_float=npY)
                            # elif evaluate_type == "train_and_save":
                            #     model = train_model(npX, npY_class)
                            #     save_model(model, model_file_prefix)
                            # elif evaluate_type == "test_trained":
                            #     evaluate(len(np.unique(npY_class)), npX, npY_class, model_file_prefix)

                finally:
                    en.notify()
                    plt.show()