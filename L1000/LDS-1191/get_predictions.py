import csv
import datetime
import json
from pathlib import Path

import numpy as np
from L1000.data_loader import get_feature_dict, load_csv
from L1000.gene_predictor import load_model
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import helpers.email_notifier as en
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
set_session(tf.Session(config=config))
import time
from multiprocessing import Pool
import multiprocessing as mp
from contextlib import closing

down_model_file_prefixes = [
    'PC3Down5bin-1k77',
    'VCAPDown5bin-1k74',
    'A549Down5bin-1k76',
    'MCF7Down5bin-1k76',
    # 'PC35Down78',
    # 'VCAP5Down75',
    # 'HCC5155Down81',
    # 'A549Down',
    # 'HEPG2Down',
    # 'MCF7Down',
    # 'HEK293TDown',
    # 'HT29Down',
    # 'A375Down',
    # 'HA1EDown',
    # 'THP1Down',
    # 'BT20Down',
    # 'U937Down',
    # 'MCF10ADown',
    # 'HUH7Down',
    # 'NKDBADown',
    # 'NOMO1Down',
    # 'JURKATDown',
    # 'SKBR3Down',
    # 'HS578TDown',
    # 'MDAMB231Down'
]

up_model_file_prefixes = [
    'PC3Up5bin-1k78',
    'VCAPUp5bin-1k73',
    'A549Up5bin-1k77',
    'MCF7Up5bin-1k77',
    # 'PC35Up77',
    # 'VCAP5Up74',
    # 'A5495Up.75',
    # 'HCC5155Up.80',
    # 'HEPG2Up',
    # 'MCF7Up',
    # 'HEK293TUp',
    # 'HT29Up',
    # 'A375Up',
    # 'HA1EUp',
    # 'THP1Up',
    # 'BT20Up',
    # 'U937Up',
    # 'MCF10AUp',
    # 'HUH7Up',
    # 'NKDBAUp',
    # 'NOMO1Up',
    # 'JURKATUp',
    # 'SKBR3Up',
    # 'HS578TUp',
    # 'MDAMB231Up'
]

ends_model_file_prefix = "VCAP_Multi10"
gene_count_data_limit = 978
find_promiscuous = True
most_promiscious_drug = ''
most_promiscious_drug_target_gene_count = 0
operation = "promiscuous"
hit_score = 0
plot_histograms = False
save_histogram_data = False
use_ends_model = False
path_prefix = "saved_models/"
# zinc_file_name  = '/home/gwoo/Data/zinc/ZincCompounds_InStock_maccs.tab'
# zinc_file_name  = 'data/german_smiles_rdkit_maccs.csv'
# zinc_file_name  = 'data/nathan_smiles_rdkit_maccs.csv'
# zinc_file_name  = 'data/smiles_rdkit_maccs.csv'

class Top10():
    def __init__(self):
        self.max_size = 20
        self.current_size = 0
        self.sorted_dict = SortedDict()

    def add_item(self, key, value):
        if self.current_size >= self.max_size:
            self.sorted_dict.popitem(False)
        else:
            self.current_size += 1
        ctr = 0
        key_suffix = str(key).zfill(3) + "_" + str(ctr)
        while key_suffix in self.sorted_dict:
            ctr += 1
            key_suffix = str(key).zfill(3) + "_" + str(ctr)
        self.sorted_dict[key_suffix] = value

    def get_lowest_key_prefix(self):
        key = self.sorted_dict.keys()[0]
        tokens = key.split('_')
        return int(tokens[0])

    def get_dict(self):
        return self.sorted_dict

class Top10Float(Top10):
    def add_item(self, key, value):
        if self.current_size >= self.max_size:
            self.sorted_dict.popitem(False)
        else:
            self.current_size += 1
        key_suffix = str(key)
        self.sorted_dict[key_suffix] = value

    def get_lowest_key_prefix(self):
        key = self.sorted_dict.keys()[0]
        return float(key)

# load model
def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)

def get_gene_id_dict():
    lm_genes = json.load(open('data/lm_plus_ar_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict
gene_id_dict = get_gene_id_dict()

def save_list(list, direction, iteration):
    prefix = "save_counts/"
    file = open(prefix + direction + iteration + '.txt', 'w')
    for item in list:
        file.write(str(item) + "\n")

def ndprint(a, format_string='{0:.3f}'):
    print([format_string.format(val, i) for i, val in enumerate(a)])

def get_specific_score(num_pert_samples, up_samples, down_samples, flat_samples, model, up_model, down_model):
    pert_sum = 0.0
    n_up = len(up_samples)
    n_down = len(down_samples)
    n_flat = len(flat_samples)
    all_samples = flat_samples
    if n_up > 0:
        all_samples = np.concatenate((all_samples, up_samples))
    if n_down > 0:
        all_samples = np.concatenate((all_samples, down_samples))
    up_start = n_flat
    down_start = n_flat + n_up

    if use_ends_model:
        all_predictions = model.predict(all_samples)
    else:
        up_model_all_predictions = up_model.predict(all_samples)
        down_model_all_predictions = down_model.predict(all_samples)
        flipped_down_predictions = np.fliplr(down_model_all_predictions)
        all_predictions = np.mean(np.array([up_model_all_predictions, flipped_down_predictions]), axis=0)

    ndprint(all_predictions[:, 0])
    ndprint(all_predictions[:, 1])
    ndprint(all_predictions[:, 2])
    flat_predictions = all_predictions[:up_start - 1]
    up_predictions = all_predictions[up_start: down_start - 1]
    down_predictions = all_predictions[down_start:]

    for prediction in up_predictions:
        pert_sum += prediction[1]
    for prediction in down_predictions:
        pert_sum += prediction[0]
    if num_pert_samples <= 0:
        pert_score = 0
    else:
        pert_score = pert_sum / num_pert_samples

    flat_sum = 0.0
    for prediction in flat_predictions:
        if prediction[1] > 0.5:
            flat_sum += prediction[1]
        else:
            flat_sum += prediction[0]
    flat_score = flat_sum / n_flat
    return pert_score, flat_score, pert_score / flat_score

def get_specific_score_concensus(num_pert_samples, up_samples, down_samples, flat_samples, model, up_models, down_models):
    num_models = max(len(up_models), 1)
    pert_score_sum = 0
    flat_score_sum = 0
    if use_ends_model:
        pert_score_sum, flat_score_sum, total_score = get_specific_score(num_pert_samples, up_samples, down_samples,
                                                                 flat_samples, model, None, None)
    else:
        for i in range(0, num_models):
            up_model = up_models[i]
            down_model = down_models[i]
            pert_score, flat_score, total_score = get_specific_score(num_pert_samples, up_samples, down_samples,
                                                                 flat_samples, model, up_model, down_model)
            pert_score_sum += pert_score
            flat_score_sum += flat_score
    avg_pert_score = pert_score_sum / num_models
    avg_flat_score = flat_score_sum / num_models

    return avg_pert_score, avg_flat_score, avg_pert_score / avg_flat_score

def get_specific_predictions(up_gene_ids, down_gene_ids, score_function, model, up_models, down_models):

    def get_drug_features(row):
        drug_features = []
        for value in row[1:]:
            drug_features.append(int(float(value)))
        return drug_features

    def get_samples(drug_features, gene_features_list):
        samples_list = []
        for gene_features in gene_features_list:
            samples_list.append(np.asarray(drug_features + gene_features))
        return np.asarray(samples_list, dtype="float16")

    def get_genes_features_list(up_gene_ids, down_gene_ids):
        gene_features_dict = get_feature_dict('data/lm_ar_gene_go_fingerprint.csv', use_int=True)
        gene_ids_by_var = load_csv('data/genes_by_var_lm_ar.csv')

        gene_ids_list = []
        for sublist in gene_ids_by_var:
            for item in sublist:
                gene_ids_list.append(item)

        gene_name_list = []
        up_gene_features_list = []
        down_gene_features_list = []
        flat_gene_features_list = []
        for gene_id in gene_ids_list:
            gene_symbol = gene_id_dict[gene_id]
            if gene_symbol not in gene_features_dict:
                continue
            gene_features = gene_features_dict[gene_symbol]

            if gene_id in up_gene_ids:
                up_gene_features_list.append(gene_features)
            elif gene_id in down_gene_ids:
                down_gene_features_list.append(gene_features)
            else:
                # print('flat gene id', gene_id)
                flat_gene_features_list.append(gene_features)
                gene_name_list.append(gene_symbol)
        print(gene_name_list)
        return up_gene_features_list, down_gene_features_list, flat_gene_features_list

    top10scores = Top10Float()
    num_pert_samples = len(up_gene_ids) + len(down_gene_ids)

    up_gene_features_list, down_gene_features_list, flat_gene_features_list = get_genes_features_list(up_gene_ids,
                                                                                                      down_gene_ids)
    scores = []
    iteration = 0
    n_drugs = 500000
    n_gpus = 1
    batch_size = int(n_drugs / n_gpus)
    start = iteration * batch_size
    end = start + batch_size - 1
    print('iteration', iteration)

    with open(zinc_file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        drug_counter = 0
        for row in reader:
            try:
                if drug_counter % 100 == 0:
                    print(drug_counter)
                if save_histogram_data:
                    if drug_counter < start:
                        drug_counter += 1
                        continue
                    if drug_counter > end:
                        break

                molecule_id = row[0]
                # if molecule_id != 'ZINC000225292017':
                #     continue
                drug_features = get_drug_features(row)
                up_samples = get_samples(drug_features, up_gene_features_list)
                down_samples = get_samples(drug_features, down_gene_features_list)
                flat_samples = get_samples(drug_features, flat_gene_features_list)
                # print('mol id', molecule_id)
                num_models = max(len(up_models), 1)

                def get_scores(up_models1, down_models1):
                    pert_score, flat_score, total_score = score_function(num_pert_samples, up_samples, down_samples,
                                                                         flat_samples, model, up_models1, down_models1)

                    scores.append(total_score)
                    if top10scores.current_size < top10scores.max_size or total_score > top10scores.get_lowest_key_prefix():
                        message = "model " + str(i) + " compound " + str(molecule_id) \
                                  + " has pert score " + "{:0.4f}".format(pert_score) \
                                  + " flat score " + "{:0.4f}".format(flat_score) \
                                  + " total score " + "{:0.4f}".format(total_score)
                        top10scores.add_item(total_score, message)
                        print(datetime.datetime.now(), message)

                if use_ends_model:
                    get_scores([], [])
                else:
                    for i in range(0, num_models):
                        up_models1 = [up_models[i]]
                        down_models1 = [down_models[i]]
                        get_scores(up_models1, down_models1)

                if save_histogram_data and drug_counter % 1000 == 0:
                    save_list(scores, 'scores', str(iteration))
            except:
                continue
            finally:
                drug_counter += 1
        if save_histogram_data:
            save_list(scores, 'scores', str(iteration))

def screen_compounds():
    try:
        germans_up_genes = []
        germans_down_genes = ['6657']
        up_models = []
        down_models = []
        model = None
        if use_ends_model:
            model = load_model_from_file_prefix(path_prefix + ends_model_file_prefix)
        else:
            for i in range(0, len(down_model_file_prefixes)):
                up_models.append(load_model_from_file_prefix(path_prefix + up_model_file_prefixes[i]))
                down_models.append(load_model_from_file_prefix(path_prefix + down_model_file_prefixes[i]))

        get_specific_predictions( germans_up_genes, germans_down_genes, get_specific_score_concensus, model, up_models,
                                 down_models)
    finally:
        en.notify("Predicting Done")
        quit()

    over_expressed_genes = []
    under_expressed_genes = []

    with open("data/pca_misexpressed_lm_genes.csv", "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        for row in reader:
            if row[0] != '':
                over_expressed_genes.append(row[0])
            if row[1] != '':
                under_expressed_genes.append(row[1])
    mis_expressed_genes = over_expressed_genes + under_expressed_genes

    if use_ends_model:
        ends_model = load_model_from_file_prefix(path_prefix + ends_model_file_prefix)
        down_model = None
        up_model = None
    else:
        down_model = load_model_from_file_prefix(path_prefix + down_model_file_prefixes[0])
        up_model = load_model_from_file_prefix(path_prefix + up_model_file_prefixes[0])
        ends_model = None

    # load gene fingerprints to test
    gene_features_dict = get_feature_dict('data/gene_go_fingerprint.csv', use_int=True)
    lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
    print('gene expressions loaded. rows:  ' + str(len(lm_gene_entrez_ids_list)))
    lm_gene_entrez_ids = []
    for sublist in lm_gene_entrez_ids_list :
        for item in sublist:
            lm_gene_entrez_ids.append(item)

    top10s = {}
    top10down = Top10()
    top10up = Top10()
    top10all = Top10()
    down_counts = []
    up_counts = []
    all_counts = []

    def calculate_perturbations(model, samples, class_value, top10_list, molecule_id, direction_str, misexp_str='',
                                gene_list=[]):
        # predict the batch
        predictions = model.predict(samples)
        regulate_counter = 0
        gene_list_str = ''
        gene_counter = 0
        for prediction in predictions:
            probability = prediction[class_value]
            threshold = (0.99 if use_ends_model else 0.5)
            if probability > threshold:
                regulate_counter += 1
                if len(gene_list) > gene_counter:
                    gene_list_str += str(gene_list[gene_counter]) + " "
            gene_counter += 1

        if top10_list.current_size <= 0 or regulate_counter > top10_list.get_lowest_key_prefix():
            message = "Compound " + str(molecule_id) + " " + direction_str + "regulates " +  str(regulate_counter) + " " + \
                      misexp_str + " genes."
            message += gene_list_str
            top10_list.add_item(regulate_counter, message)
            print(datetime.datetime.now(), message)

        return regulate_counter

    def plot_histograms_func(down_counts, up_counts, all_counts):
        plt.figure()
        plt.title('Histogram of Downregulations')
        plt.ylabel('# of compounds')
        plt.xlabel('# of genes perturbated')
        plt.hist(down_counts, bins=100)
        plt.draw()

        plt.figure()
        plt.title('Histogram of Upregulations')
        plt.ylabel('# of compounds')
        plt.xlabel('# of genes perturbated')
        plt.hist(up_counts, bins=100)
        plt.draw()

        plt.figure()
        plt.title('Histogram of Number of Compounds vs Total Regulated Genes')
        plt.ylabel('# of compounds')
        plt.xlabel('# of genes perturbated')
        plt.hist(all_counts, bins=100)
        plt.draw()
        plt.show()

    try:
        iteration = 0
        n_drugs = 500000
        n_gpus = 1
        batch_size = int(n_drugs / n_gpus)
        start = iteration * batch_size
        end = start + batch_size - 1
        print('iteration', iteration)

        with open(zinc_file_name , "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel', delimiter=',' )
            next(reader)
            drug_counter = 0

            num_genes = len(lm_gene_entrez_ids)
            gene_features_list = []
            for gene_id in lm_gene_entrez_ids:
                gene_symbol = gene_id_dict[gene_id]
                if gene_symbol not in gene_features_dict:
                    continue
                gene_features = gene_features_dict[gene_symbol]
                gene_features_list.append(gene_features)

            num_over_genes = len(over_expressed_genes)
            over_expressed_gene_features_list = []
            for gene_id in over_expressed_genes:
                gene_symbol = gene_id_dict[gene_id]
                if gene_symbol not in gene_features_dict:
                    continue
                gene_features = gene_features_dict[gene_symbol]
                over_expressed_gene_features_list.append(gene_features)

            for row in reader:

                if save_histogram_data:
                    if drug_counter < start:
                        drug_counter += 1
                        continue
                    if drug_counter > end:
                        break

                molecule_id = row[0]
                if drug_counter % 10000 == 0:
                    print(datetime.datetime.now(), "Evaluating molecule #", drug_counter)
                drug_features = []
                try:
                    for value in row[1:]:
                        drug_features.append(int(value))
                except:
                    drug_counter += 1
                    continue

                # if drug_counter % 100 == 0 and plot_histograms:
                #     plot_histograms_func(down_counts, up_counts, all_counts)

                if operation == "counter":
                    # hits / (non hits + hits + missed )

                    # get the batch of samples
                    samples_batch = np.array([], dtype="f2")
                    for gene_features in over_expressed_gene_features_list:
                        samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
                    samples_batch = samples_batch.reshape([num_over_genes, -1])
                    downregulate_count = calculate_perturbations(down_model, samples_batch, 0, top10down, molecule_id,
                                                                 "down", "overexpressed", over_expressed_genes)

                    samples_batch = np.array([], dtype="f2")
                    num_under_genes = len(under_expressed_genes)
                    for gene_id in under_expressed_genes:
                        gene_symbol = gene_id_dict[gene_id]
                        if gene_symbol not in gene_features_dict:
                            continue
                        gene_features = gene_features_dict[gene_symbol]
                        samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
                    samples_batch = samples_batch.reshape([num_under_genes, -1])
                    upregulate_count = calculate_perturbations(up_model, samples_batch, 1, top10up, molecule_id,
                                                               "up", "underexpressed", under_expressed_genes)
                    allregulate_count = downregulate_count + upregulate_count
                    if top10all.current_size == 0 or \
                            (top10all.current_size > 0 and allregulate_count > top10all.get_lowest_key_prefix()):
                        message = "Compound " + str(molecule_id) + " downregulates " + str(downregulate_count) + \
                                  " overexpressed genes and upregulates " + str(upregulate_count) + \
                                  " underexpressed genes. Total: " + str(allregulate_count)
                        top10all.add_item(allregulate_count, message)
                        print(datetime.datetime.now(), message)

                elif operation == "promiscuous":
                    # get the batch of samples
                    samples_batch = np.array([], dtype="f2")
                    for gene_features in gene_features_list:
                        samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
                    samples_batch = samples_batch.reshape([num_genes, -1])

                    if use_ends_model:
                        downregulate_count = calculate_perturbations(ends_model, samples_batch, 0, top10down, molecule_id, "down")
                        upregulate_count = calculate_perturbations(ends_model, samples_batch, 1, top10up, molecule_id, "up")
                    else:
                        downregulate_count = calculate_perturbations(down_model, samples_batch, 0, top10down, molecule_id, "down")
                        upregulate_count = calculate_perturbations(up_model, samples_batch, 1, top10up, molecule_id, "up")

                    allregulate_count = downregulate_count + upregulate_count
                    if top10all.current_size == 0 or (top10all.current_size > 0 and allregulate_count > top10all.get_lowest_key_prefix()):
                        message = "Compound " + str(molecule_id) + " downregulates " + str(downregulate_count) + \
                                  " genes and upregulates " + str(upregulate_count) + " genes. Total: " + str(allregulate_count)
                        top10all.add_item(allregulate_count, message)
                        print(datetime.datetime.now(), message)
                    # else:
                    #     prediction_counter = 0
                    #     for prediction in predictions:
                    #         down_probability = prediction[0]
                    #         if down_probability > 0.5:
                    #             gene_symbol = gene_id_dict[lm_gene_entrez_ids[prediction_counter % num_genes]]
                    #             message = gene_symbol + " " + str(down_probability) + " Found compound " + str(molecule_id) \
                    #                       + " that downregulates " + gene_symbol + " " + str(down_probability)
                    #             if gene_symbol not in top10s:
                    #                 top10s[gene_symbol] = Top10()
                    #                 top10s[gene_symbol].add_item(down_probability, message)
                    #                 print(message)
                    #             else:
                    #                 if down_probability > top10s[gene_symbol].get_lowest_key():
                    #                     top10s[gene_symbol].add_item(down_probability, message)
                    #                     print(message)
                    #         prediction_counter += 1
                down_counts.append(downregulate_count)
                up_counts.append(upregulate_count)
                all_counts.append(allregulate_count)
                drug_counter += 1

        if save_histogram_data:
            save_list(down_counts, 'down', str(iteration))
            save_list(up_counts, 'up', str(iteration))
            save_list(all_counts, 'all', str(iteration))

    finally:
        en.notify("Predicting Done")
        plt.show()

def predict_arts_2():
    up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_NK_AR_Up"
    up_model = load_model_from_file_prefix(up_model_filename_prefix)
    down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_NK_AR_Down"
    down_model = load_model_from_file_prefix(down_model_filename_prefix)

    gene_features_dict = get_feature_dict('/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/ar_gene_go_fingerprint.csv')
    drug_features_dict = get_feature_dict('/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/vpc_compounds_morgan_2048_nk.csv')

    target_gene_features_dict = {
        'AR': gene_features_dict['AR'],
        'KLK3': gene_features_dict['KLK3'],
        'KLK2': gene_features_dict['KLK2'],
        'TMPRSS2': gene_features_dict['TMPRSS2'],
        'CDC20': gene_features_dict['CDC20'],
        'CDK1': gene_features_dict['CDK1'],
        'CCNA2': gene_features_dict['CCNA2'],
        'UBE2C': gene_features_dict['UBE2C'],
        'AKT1': gene_features_dict['AKT1'],
        'UGT2B15': gene_features_dict['UGT2B15'],
        'UGT2B17': gene_features_dict['UGT2B17'],
        'TRIB1': gene_features_dict['TRIB1']
    }

    data = []
    descriptions = []
    for drug in drug_features_dict:
        for gene in target_gene_features_dict:
            data.append(drug_features_dict[drug] + target_gene_features_dict[gene])
            descriptions.append(drug + ", " + gene)
    data = np.asarray(data, dtype=np.float16)


    up_predictions = up_model.predict(data)
    down_predictions = down_model.predict(data)

    for i in range(0, len(data)):
        up_prediction = up_predictions[i]
        if up_prediction[1] > 0.554:  # max f cutoff
            print(descriptions[i] + ",", "Up model, predicts, up-regulation,. Probability,", up_prediction[1])
        else:
            print(descriptions[i] + ",", "Up model, predicts, down-regulation,. Probability,", up_prediction[1])
    for i in range(0, len(data)):
        down_prediction = down_predictions[i]
        if down_prediction[1] > 0.765:  # max f cutoff
            print(descriptions[i] + ",", "Down model, predicts, down-regulation,. Probability,", down_prediction[1])
        else:
            print(descriptions[i] + ",", "Down model, predicts, up-regulation,. Probability,", down_prediction[1])

def predict_file(fname):
    ar_up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_AR_Up"
    ar_up_model = load_model_from_file_prefix(ar_up_model_filename_prefix)
    ar_down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_AR_Down"
    ar_down_model = load_model_from_file_prefix(ar_down_model_filename_prefix)
    lm_ar_up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_LM_AR_Up"
    lm_ar_up_model = load_model_from_file_prefix(lm_ar_up_model_filename_prefix)
    lm_ar_down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/VCAP_LM_AR_Down"
    lm_ar_down_model = load_model_from_file_prefix(lm_ar_down_model_filename_prefix)


    with open(fname, 'r') as file:
        count = 0
        for line in file:
            if count > 2:
                break
            print(fname, line)
            count += 1

            #
            # with open('ZINC_15_morgan_2048_2D/' + fname.split('/')[-1], 'a') as ref2:
            #     ref2.write((',').join([zin_id] + [str(elem) for elem in np.where(arg == 1)[0]]))
            #     ref2.write('\n')


def screen_zinc():
    import multiprocessing
    from multiprocessing import Pool
    from contextlib import closing
    import glob

    files = []
    for f in glob.glob('/data/datasets/gwoo/zinc/Morgan/*'):
        files.append(f)

    cpu_count = multiprocessing.cpu_count()
    cpu_count = 1
    with closing(Pool(cpu_count)) as pool:
        pool.map(predict_file, files)

def predict_nathans():
    up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/LNCAP_NK_LM_Up"
    up_model = load_model_from_file_prefix(up_model_filename_prefix)
    down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/LNCAP_NK_LM_Down"
    down_model = load_model_from_file_prefix(down_model_filename_prefix)

    gene_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/gene_go_fingerprint_moreThan3.csv')
    drug_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/nathans_morgan_2048_nk.csv')

    # target_gene_features_dict = {
    #     'AR': gene_features_dict['AR'],
    #     'KLK3': gene_features_dict['KLK3'],
    #     'KLK2': gene_features_dict['KLK2'],
    #     'TMPRSS2': gene_features_dict['TMPRSS2'],
    #     'CDC20': gene_features_dict['CDC20'],
    #     'CDK1': gene_features_dict['CDK1'],
    #     'CCNA2': gene_features_dict['CCNA2'],
    #     'UBE2C': gene_features_dict['UBE2C'],
    #     'AKT1': gene_features_dict['AKT1'],
    #     'UGT2B15': gene_features_dict['UGT2B15'],
    #     'UGT2B17': gene_features_dict['UGT2B17'],
    #     'TRIB1': gene_features_dict['TRIB1']
    # }

    data = []
    descriptions = []
    for drug in drug_features_dict:
        for gene in gene_features_dict:
            data.append(drug_features_dict[drug] + gene_features_dict[gene])
            descriptions.append(drug + " " + gene)
    data = np.asarray(data, dtype=np.float16)

    up_predictions = up_model.predict(data)
    down_predictions = down_model.predict(data)

    for i in range(0, len(data)):
        up_prediction = up_predictions[i]
        if up_prediction[1] > 0.561:  # max f cutoff
            print(descriptions[i], "Up Probability", up_prediction[1])
    for i in range(0, len(data)):
        down_prediction = down_predictions[i]
        if down_prediction[1] > 0.649:  # max f cutoff
            print(descriptions[i], "Down Probability", down_prediction[1])


def get_target_score(num_pert_samples, up_samples, down_samples, flat_samples, model, up_models, down_models):
    up_auc = 0.823
    down_auc = 0.827
    up_weight = up_auc / (up_auc + down_auc)
    down_weight = 1 - up_weight
    n_up = len(up_samples)
    n_down = len(down_samples)

    up_model_all_predictions = up_models[0].predict(up_samples)
    down_model_all_predictions = down_models[0].predict(down_samples)

    average_up_gene_probability = np.sum(up_model_all_predictions[:, 1]) / n_up
    average_down_gene_probability = np.sum(down_model_all_predictions[:, 1]) / n_down

    pert_score = up_weight * average_up_gene_probability + down_weight * average_down_gene_probability

    return pert_score, 0.0, pert_score


def get_predictions(zinc_file_name, up_gene_ids, down_gene_ids, up_model, down_model):

    def get_drug_features(row):
        # drug_features = []
        # for value in row[1:]:
        #     drug_features.append(int(float(value)))
        # return drug_features
        introw = []
        for item in row[1:]:
            introw.append(int(item))
        drug_features = np.zeros(2048)
        drug_features[introw] = 1
        return drug_features.tolist()

    def get_samples(drug_features, gene_features_list):
        samples_list = []
        for gene_features in gene_features_list:
            samples_list.append(np.asarray(drug_features + gene_features))
        return np.asarray(samples_list, dtype="float16")

    def get_genes_features_list(up_gene_ids, down_gene_ids):
        gene_features_dict = get_feature_dict('data/lm_ar_gene_go_fingerprint.csv', use_int=True)
        gene_ids_by_var = load_csv('data/genes_by_var_lm_ar.csv')

        gene_ids_list = []
        for sublist in gene_ids_by_var:
            for item in sublist:
                gene_ids_list.append(item)

        gene_name_list = []
        up_gene_features_list = []
        down_gene_features_list = []
        flat_gene_features_list = []
        for gene_id in gene_ids_list:
            gene_symbol = gene_id_dict[gene_id]
            if gene_symbol not in gene_features_dict:
                continue
            gene_features = gene_features_dict[gene_symbol]

            if gene_id in up_gene_ids:
                up_gene_features_list.append(gene_features)
            elif gene_id in down_gene_ids:
                down_gene_features_list.append(gene_features)
            else:
                # print('flat gene id', gene_id)
                flat_gene_features_list.append(gene_features)
                gene_name_list.append(gene_symbol)
        # print(gene_name_list)
        return up_gene_features_list, down_gene_features_list, flat_gene_features_list

    def get_multi_mol_target_score(n_mol, up_samples, down_samples, up_model, down_model, n_target_genes):
        up_auc = 0.823
        down_auc = 0.827
        up_weight = up_auc / (up_auc + down_auc)
        down_weight = 1 - up_weight
        n_up = len(up_samples)
        n_down = len(down_samples)

        up_model_all_predictions = up_model.predict(up_samples)
        down_model_all_predictions = down_model.predict(down_samples)

        n_up_samples_per_mol = int(n_up / n_mol)
        n_down_samples_per_mol = int(n_down / n_mol)

        pert_scores = []
        for i in range(0, n_mol):
            up_start = i * n_up_samples_per_mol
            up_end = (i + 1) * n_up_samples_per_mol - 1
            up_gene_probability_sum = np.sum(up_model_all_predictions[up_start:up_end, 1]) * up_weight
            down_start = i * n_down_samples_per_mol
            down_end = (i + 1) * n_down_samples_per_mol - 1
            down_gene_probability_sum = np.sum(down_model_all_predictions[down_start:down_end, 1]) * down_weight
            pert_score = (up_gene_probability_sum + down_gene_probability_sum) / n_target_genes
            pert_scores.append(pert_score)

        return pert_scores

    def predict_batch(mol_id_batch, up_samples_batch, down_samples_batch):
        score_list = []
        nd_up_samples_batch = np.asarray(up_samples_batch)
        nd_down_samples_batch = np.asarray(down_samples_batch)
        pert_score_batch = get_multi_mol_target_score(len(mol_id_batch), nd_up_samples_batch,
                                                      nd_down_samples_batch, up_model, down_model,
                                                      n_genes)
        for i in range(0, len(mol_id_batch)):
            # get one score per compound
            molecule_id_inner = mol_id_batch[i]
            pert_score = pert_score_batch[i]
            scores.append(pert_score)
            # if top10scores.current_size < top10scores.max_size or \
            #         pert_score > top10scores.get_lowest_key_prefix():
            message = str(molecule_id_inner) + ", " + "{:0.4f}".format(pert_score)
            # top10scores.add_item(pert_score, message)
            score_list.append(message)
            print(datetime.datetime.now(), message)
        return score_list
    # top10scores = Top10Float()

    up_gene_features_list, down_gene_features_list, flat_gene_features_list = get_genes_features_list(up_gene_ids,
                                                                                                      down_gene_ids)
    n_genes = len(up_gene_features_list) + len(down_gene_features_list)
    scores = []
    iteration = 0
    n_drugs = 500000
    n_gpus = 1
    batch_size = int(n_drugs / n_gpus)
    start = iteration * batch_size
    end = start + batch_size - 1
    print('iteration', iteration)
    mols_per_batch = 10000
    start_time = time.time()
    toSave = []

    with open(zinc_file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        drug_counter = 1
        mol_id_batch = []
        up_samples_batch = []
        down_samples_batch = []
        for row in reader:
            try:
                if drug_counter % 100 == 0:
                    print(drug_counter, time.time() - start_time)
                if save_histogram_data:
                    if drug_counter < start:
                        drug_counter += 1
                        continue
                    if drug_counter > end:
                        break
                molecule_id = row[0]
                drug_features = get_drug_features(row)
                up_samples = get_samples(drug_features, up_gene_features_list)
                down_samples = get_samples(drug_features, down_gene_features_list)
                # print('mol id', molecule_id)

                mol_id_batch.append(molecule_id)
                up_samples_batch.extend(up_samples)
                down_samples_batch.extend(down_samples)

                if drug_counter % mols_per_batch == 0:
                    score_list = predict_batch(mol_id_batch, up_samples_batch, down_samples_batch)
                    toSave.extend(score_list)
                    mol_id_batch = []
                    up_samples_batch = []
                    down_samples_batch = []
            except:
                continue
            finally:
                drug_counter += 1
        score_list = predict_batch(mol_id_batch, up_samples_batch, down_samples_batch)
        toSave.extend(score_list)
        with open(zinc_file_name + '.scores.csv', 'w') as f:
            for item in toSave:
                f.write("%s\n" % item)


def screen_for_ar_compounds(file):
    ar_up_genes = ['7366', '7367', '10221']
    ar_down_genes = ['367', '354', '3817', '7113', '991', '983', '890', '11065', '207']
    path_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/"
    up_model_file_prefix = "VCAP_NK_LM_AR_Up"
    down_model_file_prefix = "VCAP_NK_LM_AR_Down"
    try:
        up_model = load_model_from_file_prefix(path_prefix + up_model_file_prefix)
        down_model = load_model_from_file_prefix(path_prefix + down_model_file_prefix)
        # get_specific_predictions(ar_up_genes, ar_down_genes, get_target_score, None, [up_model], [down_model])
        get_predictions(file, ar_up_genes, ar_down_genes, up_model, down_model)
    finally:
        en.notify("Predicting Done" + file)


def unpack_get_predictions(args):
    file = args[0]
    ar_up_genes = args[1]
    ar_down_genes = args[2]
    up_model_file_prefix = args[3]
    down_model_file_prefix = args[4]
    up_model = load_model_from_file_prefix(up_model_file_prefix)
    down_model = load_model_from_file_prefix(down_model_file_prefix)
    get_predictions(file, ar_up_genes, ar_down_genes, up_model, down_model)


def split_multi_process():
    data_path = 'data/'
    files = os.listdir(data_path)
    smi_files = []
    for file in files:
        if file.endswith('.smi') and not file == 'merged_id_smiles.smi':
            smi_files.append(file)

    # for file in smi_files:
    #     screen_for_ar_compounds(data_path + file)

    try:
        with closing(Pool(mp.cpu_count())) as pool:
            pool.map(screen_for_ar_compounds, smi_files)
    finally:
        en.notify("Predicting Done All Files")


# screen_compounds()
# predict_arts_2()
# screen_zinc()
# predict_nathans()
# screen_for_ar_compounds('data/non_kekulized_morgan_2048.csv')
split_multi_process()
