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

down_model_file_prefix = "100PC3Down"
up_model_file_prefix = "100PC3Up"
ends_model_file_prefix = "PC3EndsModel85"
gene_count_data_limit = 978
find_promiscuous = True
most_promiscious_drug = ''
most_promiscious_drug_target_gene_count = 0
operation = "promiscuous"
hit_score = 0
plot_histograms = False
save_histogram_data = True
use_ends_model = True
path_prefix = "saved_models/"
zinc_file_name  = '/home/gwoo/Data/zinc/ZincCompounds_InStock_maccs.tab'
# zinc_file_name  = 'data/nathan_smiles_rdkit_maccs.csv'

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
    lm_genes = json.load(open('data/landmark_genes.json'))
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

def get_specific_predictions(up_gene_ids, down_gene_ids):

    def get_drug_features(row):
        drug_features = []
        for value in row[1:]:
            drug_features.append(int(value))
        return drug_features

    def get_samples(drug_features, gene_features_list):
        num_genes = len(gene_features_list)
        samples_batch = np.array([], dtype="float16")
        for gene_features in gene_features_list:
            samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
        return samples_batch.reshape([num_genes, -1])

    def get_genes_features_list(up_gene_ids, down_gene_ids):
        gene_features_dict = get_feature_dict('data/gene_go_fingerprint.csv', use_int=True)
        gene_ids_by_var = load_csv('data/genes_by_var.csv')

        gene_ids_list = []
        for sublist in gene_ids_by_var:
            for item in sublist:
                gene_ids_list.append(item)

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
                flat_gene_features_list.append(gene_features)
        return up_gene_features_list, down_gene_features_list, flat_gene_features_list

    def get_specific_score(num_pert_samples, up_samples, down_samples, flat_samples, model):
        predictions = model.predict(up_samples)
        pert_sum = 0.0
        for prediction in predictions:
            pert_sum += prediction[1]
        predictions = model.predict(down_samples)
        for prediction in predictions:
            pert_sum += prediction[0]
        pert_score = pert_sum / num_pert_samples
        predictions = model.predict(flat_samples)
        num_flat_samples = len(predictions)
        flat_sum = 0.0
        for prediction in predictions:
            if prediction[1] > 0.5:
                flat_sum += prediction[1]
            else:
                flat_sum += prediction[0]
        flat_score = 1 - flat_sum / num_flat_samples
        return pert_score, flat_score, pert_score + flat_score

    top10scores = Top10Float()
    num_pert_samples = len(up_gene_ids) + len(down_gene_ids)
    model = load_model_from_file_prefix(path_prefix + ends_model_file_prefix)
    up_gene_features_list, down_gene_features_list, flat_gene_features_list = get_genes_features_list(up_gene_ids,
                                                                                                      down_gene_ids)
    scores = []
    iteration = 0
    n_drugs = 500000
    n_gpus = 14
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
                flat_samples = get_samples(drug_features, flat_gene_features_list)
                pert_score, flat_score, total_score = get_specific_score(num_pert_samples, up_samples, down_samples,
                                                                         flat_samples, model)
                scores.append(total_score)
                if top10scores.current_size == 0 or \
                        (top10scores.current_size > 0 and total_score > top10scores.get_lowest_key_prefix()):
                    message = "compound " + str(molecule_id) \
                              + " has pert score " + "{:0.4f}".format(pert_score) \
                              + " flat score " + "{:0.4f}".format(flat_score) \
                              + " total score " + "{:0.4f}".format(total_score)
                    top10scores.add_item(total_score, message)
                    print(datetime.datetime.now(), message)

                if save_histogram_data and drug_counter % 1000 == 0:
                    save_list(scores, 'scores', str(iteration))
            except:
                continue
            finally:
                drug_counter += 1

try:
    germans_up_genes = ['7852', '3815']
    germans_down_genes = ['6657', '652']
    get_specific_predictions(germans_up_genes, germans_down_genes)
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
    down_model = load_model_from_file_prefix(path_prefix + down_model_file_prefix)
    up_model = load_model_from_file_prefix(path_prefix + up_model_file_prefix)
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