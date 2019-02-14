import datetime
import gc
import json
import time

import matplotlib.pyplot as plt
# from random_forest import do_optimize
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv
from L1000.gene_predictor import train_model, save_model

import helpers.email_notifier as en
from mlp_optimizer import do_optimize

start_time = time.time()
gene_count_data_limit = 978
use_optimizer = False
model_file_prefix = "test"

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
drug_features_dict = get_feature_dict('LDS-1484/data/LDS1484_compounds_morgan_2048_nk.csv')
# drug_features_dict = get_feature_dict('LDS-1191/data/non_kekulized_morgan_2048.csv')
# gene_features_dict = get_feature_dict('LDS-1191/data/lm_ar_gene_go_fingerprint.csv', use_int=True)
gene_features_dict = get_feature_dict('LDS-1191/data/gene_go_fingerprint_moreThan3.csv', use_int=True)
# info to separate by data by cell lines, drug + gene tests may not be equally spread out across cell lines
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1484/Metadata/Cell_Line_Metadata.txt', '\t', 2)
# info to remove any dosages that are not 'µM'. Want to standardize the dosages.
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1484/Metadata/GSE70138_Broad_LINCS_sig_info.txt', '\t', 0)
load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1484/load_data/morgan2048/"
target_cell_names = ["LNCAP"]
directions = ['Down', 'Up']

def get_their_id(good_id):
    return 'b\'' + good_id + '\''

def get_our_id(bad_id):
    return bad_id[2:-1]

# getting the gene ids
gene_id_dict = get_gene_id_dict()
# lm_gene_entrez_ids = list(gene_id_dict.keys())[:200]
lm_gene_entrez_ids_list = load_csv('LDS-1484/data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(get_their_id(item))

print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1484/Data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)

cell_X = {}
cell_Y = {}
cell_Y_gene_ids = {}
unique_cell_drugs = {}
repeat_X = {}
gene_perts = {}

use_gene_specific_cutoffs = True

for gene_id in lm_gene_entrez_ids_list:
    gene_perts[gene_id[0]] = []
test_dict = {}
# For every experiment
print("Loading experiments")
one_percent = int(length/50)
bins = [10]
for bin in bins:
    for i in range(length-1, -1, -1): # go backwards, assuming later experiments have stronger perturbation
        if i % one_percent == 0:
            printProgressBar(length - i, length, prefix='Progress:', suffix='Complete', length=50)
        X = []
        Y = []

        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]

        # parse the time
        start = col_name.rfind("_")
        end = find_nth(col_name, ":", 1)
        exposure_time = col_name[start + 1:end]
        # if exposure_time != "6H": # column counts: 6h 95219, 24h 109287, 48h 58, 144h 1
        if exposure_time != "24H":
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
        dose_unit = experiment_data[4][-2:]
        if dose_unit != 'um': # standardize dose amounts
            # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
            continue
        dose_amt = float(experiment_data[4][:-2])
        # if dose_amt < bin - 0.1 or dose_amt > bin + 0.1:  # only use the 5 mm bin
        #     continue
        # if dose_amt > 5 or dose_amt < 3:
        if dose_amt < 10:
            continue

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]
        # cell_name = "A375"  # this line will combine all cell lines into one
        if cell_name not in target_cell_names:
            continue

        if cell_name not in cell_name_to_id_dict:
            continue
        cell_id = cell_name_to_id_dict[cell_name][0]

        print(dose_amt, exposure_time)

        for gene_id in lm_gene_entrez_ids_list:
            our_gene_id = gene_id[0]
            gene_symbol = gene_id_dict[our_gene_id]

            if gene_symbol not in gene_features_dict:
                continue

            repeat_key = drug_id + "_" + cell_id + "_" + our_gene_id
            # if repeat_key in repeat_X and dose_amt <= repeat_X[repeat_key]:
            #     # print("repeat_key", repeat_key, "dose amount", dose_amt, "is less than", repeat_X[repeat_key])
            #     continue
            # repeat_X[repeat_key] = dose_amt

            pert = column[get_their_id(gene_id[0])]

            abspert = abs(pert)
            repeat_key = drug_id + "_" + cell_id + "_" + our_gene_id
            if repeat_key in repeat_X and abspert <= repeat_X[repeat_key]:
                # print("found duplicate repeat key", repeat_key)
                continue

            updated = False
            # repeat_X[repeat_key] = pert_conc_ratio
            if repeat_key in repeat_X:
                updated = True
            repeat_X[repeat_key] = abspert

            if cell_id not in cell_X:
                cell_X[cell_id] = {}
                cell_Y[cell_id] = {}
                unique_cell_drugs[cell_id] = {}
                cell_Y_gene_ids[cell_id] = []

            if gene_count_data_limit > 1:
                cell_X[cell_id][repeat_key] = drug_features + gene_features_dict[gene_symbol]
            else:
                cell_X[cell_id][repeat_key] = drug_features

            if repeat_key not in cell_Y[cell_id]:
                cell_Y[cell_id][repeat_key] = []
            cell_Y[cell_id][repeat_key].append(pert)
            gene_perts[our_gene_id].append(pert)
            if not updated:
                if drug_id not in unique_cell_drugs[cell_id]:
                    unique_cell_drugs[cell_id][drug_id] = 1
                cell_Y_gene_ids[cell_id].append(our_gene_id)

elapsed_time = time.time() - start_time
print("Time to load data:", elapsed_time)

gene_cutoffs_down = {}
gene_cutoffs_up = {}
percentile_down = 5 # for downregulation, use 95 for upregulation
percentile_up = 95
percentile = 5 # for downregulation, use 95 for upregulation
prog_ctr = 0
for gene_id in lm_gene_entrez_ids_list:
    prog_ctr += 1
    printProgressBar(prog_ctr, gene_count_data_limit, prefix='Storing percentile cutoffs')
    our_gene_id = gene_id[0]
    their_gene_id = get_their_id(our_gene_id)
    row = level_5_gctoo.data_df.loc[their_gene_id, :].values
    gene_cutoffs_down[our_gene_id] = np.percentile(row, percentile_down)
    gene_cutoffs_up[our_gene_id] = np.percentile(row, percentile_up)

gc.collect()
cell_line_counter = 1
gap_factor = 0.0
print("Gene count:", gene_count_data_limit, "\n")
try:
    for cell_name in cell_name_to_id_dict:
        cell_id = cell_name_to_id_dict[cell_name][0]
        if cell_id not in cell_X:
            continue
        print(datetime.datetime.now(), "Converting dictionary values to np")
        npX = np.asarray(list(cell_X[cell_id].values()), dtype='float16')
        y_pert_lists = np.asarray(list(cell_Y[cell_id].values()))
        npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])
        n_samples = len(y_pert_lists)
        npY_class = np.zeros(n_samples, dtype=int)

        for direction in directions:

            model_file_prefix = load_data_folder_path + cell_name + '_' + direction + '_' + \
                                str(bins[0]) + 'b_' + str(percentile_down) + 'p'
            print(model_file_prefix)
            prog_ctr = 0
            combined_locations = []
            combined_test_locations = []
            for gene_id in lm_gene_entrez_ids:  # this section is for gene specific class cutoffs
                prog_ctr += 1
                printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
                our_gene_id = get_our_id(gene_id)
                class_cut_off_down = gene_cutoffs_down[our_gene_id]
                class_cut_off_up = gene_cutoffs_up[our_gene_id]
                gene_locations = np.where(npY_gene_ids == our_gene_id)
                down_threshold = class_cut_off_down  # - abs(gap_factor * class_cut_off_down)
                up_threshold = class_cut_off_up  # + abs(gap_factor * class_cut_off_up)
                mid_threshold_bottom = class_cut_off_down + abs(gap_factor * class_cut_off_down)
                mid_threshold_top = class_cut_off_up - abs(gap_factor * class_cut_off_up)

                def get_class_vote(pert_list, bottom_threshold, top_threshold,
                                   mid_bottom_threshold, mid_top_threshold):
                    votes = [0, 0, 0, 0]
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
                else:  # direction = multi
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
                npY_class_save = npY_class[combined_locations]
            print("Evaluating cell line", cell_line_counter, cell_name, "(Percentile ends:", percentile_down, ")")

            sample_size = len(npY_class)

            # if sample_size < 300: # smaller sizes was giving y values of only one class
            #     continue

            print('Positive samples', np.sum(npY_class))

            num_drugs = len(unique_cell_drugs[cell_id])
            print("Sample Size:", sample_size, "Drugs tested:", num_drugs)

            np.savez(model_file_prefix + "_24h_10um_repeat_npX", npX)
            np.savez(model_file_prefix + "_24h_10um_repeat_npY_class", npY_class)

            # for drug in unique_cell_drugs[cell_id]:
            #     print(drug)

    # for cell_name in cell_name_to_id_dict:
    #     # print("Looking at", cell_name)
    #     cell_id = cell_name_to_id_dict[cell_name][0]
    #     if cell_id not in cell_X:
    #         # print("Skipping", cell_name, ". No cell line data.\n")
    #         continue
    #     listX = []
    #     listY = []
    #     listKeys = []
    #     for key, value in cell_X[cell_id].items():
    #         listX.append(value)
    #         listKeys.append(key.split("_")[0])
    #     for key, value in cell_Y[cell_id].items():
    #         listY.append(value)
    #
    #     npX = np.asarray(listX, dtype='float16')
    #     npY = np.asarray(listY, dtype='float16')
    #     npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])
    #
    #     npY_class = np.zeros(len(npY), dtype=int)
    #     if use_gene_specific_cutoffs:
    #         prog_ctr = 0
    #         combined_locations = []
    #         for their_gene_id in lm_gene_entrez_ids_list: # this section is for gene specific class cutoffs
    #             gene_id = their_gene_id[0]
    #             prog_ctr += 1
    #             printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
    #             class_cut_off_down = gene_cutoffs_down[gene_id]
    #             class_cut_off_up = gene_cutoffs_up[gene_id]
    #             gene_locations = np.where(npY_gene_ids == gene_id)
    #             down_locations = np.where(npY <= class_cut_off_down)
    #             up_locations = np.where(npY >= class_cut_off_up)
    #             npY_class[up_locations] = 1
    #             intersect = np.intersect1d(gene_locations, down_locations)
    #             combined_locations += intersect.tolist()
    #             intersect = np.intersect1d(gene_locations, up_locations)
    #             combined_locations += intersect.tolist()
    #         npX = npX[combined_locations]
    #         npY_class = npY_class[combined_locations]
    #         print("size after top down split", len(npY_class))
    #         print("Evaluating cell line", cell_line_counter, cell_name, "(Percentile ends:", percentile_down, ")")
    #     else:
    #         npY_class[np.where(npY > class_cut_off_down)] = 1 # generic class cutoff
    #         print("Evaluating cell line", cell_line_counter, cell_name, "class cutoff", class_cut_off_down, "(Percentile ends:", percentile_down, ")")
    #
    #     sample_size = len(npY_class)
    #
    #     if sample_size < 300: # smaller sizes was giving y values of only one class
    #         # print("Skipping", cell_name, ". Sample size", sample_size, "is too small.\n")
    #         continue
    #
    #     num_drugs = len(set(cell_drugs[cell_id]))
    #     print("Sample Size:", sample_size, "Drugs tested:", num_drugs)
    #
    #     # count number of perturbed genes
    #     # key_set = set(listKeys)
    #     # set_keys = np.array(listKeys)
    #     # for drug_key in key_set:
    #     #     drug_idx = np.where(set_keys == drug_key)[0].tolist()
    #     #     print(drug_key, gene_count_data_limit-np.count_nonzero(npY_class[drug_idx]))
    #
    #     if use_optimizer:
    #         do_optimize(2, npX, npY_class) #, listKeys)
    #     else:
    #         model = train_model(npX, npY_class)
    #         save_model(model, model_file_prefix)
finally:
    en.notify()
    plt.show()
