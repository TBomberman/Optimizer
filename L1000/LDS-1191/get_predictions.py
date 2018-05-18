import csv
import datetime
import json
from pathlib import Path

import numpy as np
from L1000.data_loader import get_feature_dict, load_csv
from L1000.gene_predictor import load_model
from sortedcontainers import SortedDict

import helpers.email_notifier as en

down_model_file_prefix = "100PC3Down"
up_model_file_prefix = "100PC3Up"
gene_count_data_limit = 100
find_promiscuous = True
most_promiscious_drug = ''
most_promiscious_drug_target_gene_count = 0
operation = "counter"
hit_score = 0

class Top10():
    def __init__(self):
        self.max_size = 10
        self.current_size = 0
        self.sorted_dict = SortedDict()

    def add_item(self, key, value):
        if self.current_size >= self.max_size:
            self.sorted_dict.popitem(False)
        else:
            self.current_size += 1
        self.sorted_dict[key] = value

    def get_lowest_key(self):
        return self.sorted_dict.keys()[0]

    def get_dict(self):
        return self.sorted_dict

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

# load model
def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file + "File not found")
    return load_model(model_file_prefix)

down_model = load_model_from_file_prefix(down_model_file_prefix)
up_model = load_model_from_file_prefix(up_model_file_prefix)

# load gene fingerprints to test
gene_features_dict = get_feature_dict('data/gene_go_fingerprint.csv', use_int=True)
lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)
def get_gene_id_dict():
    lm_genes = json.load(open('data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict
gene_id_dict = get_gene_id_dict()

file_name = '/home/gwoo/Data/zinc/ZincCompounds_InStock_maccs.tab'
top10s = {}
top10down = Top10()
top10up = Top10()
top10all = Top10()

def calculate_perturbations(model, samples, class_value, top10_list, molecule_id, direction_str, misexp_str='',
                            gene_list=[]):
    # predict the batch
    predictions = model.predict(samples)
    regulate_counter = 0
    gene_list_str = ''
    gene_counter = 0
    for prediction in predictions:
        probability = prediction[class_value]
        if probability > 0.5:
            regulate_counter += 1
            if len(gene_list) > gene_counter:
                gene_list_str += str(gene_list[gene_counter]) + " "
        gene_counter += 1

    if top10_list.current_size == 0 or (top10_list.current_size > 0 and regulate_counter > top10_list.get_lowest_key()):
        message = "Compound " + str(molecule_id) + " " + direction_str + "regulates " +  str(regulate_counter) + " " + \
                  misexp_str + " genes."
        message += gene_list_str
        top10_list.add_item(regulate_counter, message)
        print(datetime.datetime.now(), message)

    return regulate_counter

try:
    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        drug_counter = 0
        for row in reader:
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

            if operation == "counter":
                # hits / (non hits + hits + missed)

                # get the batch of samples
                samples_batch = np.array([], dtype="f2")
                num_over_genes = len(over_expressed_genes)
                for gene_id in over_expressed_genes:
                    gene_symbol = gene_id_dict[gene_id]
                    if gene_symbol not in gene_features_dict:
                        continue
                    gene_features = gene_features_dict[gene_symbol]
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
                        (top10all.current_size > 0 and allregulate_count > top10all.get_lowest_key()):
                    message = "Compound " + str(molecule_id) + " downregulates " + str(downregulate_count) + \
                              " overexpressed genes and upregulates " + str(upregulate_count) + \
                              " underexpressed genes. Total: " + str(allregulate_count)
                    top10all.add_item(allregulate_count, message)
                    print(datetime.datetime.now(), message)

            elif operation == "promiscuous":
                # get the batch of samples
                samples_batch = np.array([], dtype="f2")
                num_genes = len(lm_gene_entrez_ids)
                for gene_id in lm_gene_entrez_ids:
                    gene_symbol = gene_id_dict[gene_id]
                    if gene_symbol not in gene_features_dict:
                        continue
                    gene_features = gene_features_dict[gene_symbol]
                    samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
                samples_batch = samples_batch.reshape([num_genes, -1])

                downregulate_count = calculate_perturbations(down_model, samples_batch, 0, top10down, molecule_id, "down")
                upregulate_count = calculate_perturbations(up_model, samples_batch, 1, top10up, molecule_id, "up")
                allregulate_count = downregulate_count + upregulate_count
                if top10all.current_size == 0 or (top10all.current_size > 0 and allregulate_count > top10all.get_lowest_key()):
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

            drug_counter += 1
finally:
    en.notify("Predicting Done")