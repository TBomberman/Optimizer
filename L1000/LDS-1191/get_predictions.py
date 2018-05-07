import csv
import datetime
import json
from pathlib import Path

import numpy as np
from L1000.data_loader import get_feature_dict, load_csv
from L1000.gene_predictor import load_model
from sortedcontainers import SortedDict

import helpers.email_notifier as en

model_file_prefix = "100PC3"
gene_count_data_limit = 100

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

# load model
model = object
model_file = Path(model_file_prefix + ".json")
if not model_file.is_file():
    print(model_file + "File not found")
model = load_model(model_file_prefix)

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

try:
    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        drug_counter = 0
        for row in reader:
            if drug_counter % 10000 == 0:
                print(datetime.datetime.now(), "Evaluating molecule #", drug_counter)
            drug_features = []
            try:
                for value in row[1:]:
                    drug_features.append(int(value))
            except:
                drug_counter += 1
                continue

            # get the batch of samples
            samples_batch =np.array([])
            num_genes = len(lm_gene_entrez_ids)
            for gene_id in lm_gene_entrez_ids:
                gene_symbol = gene_id_dict[gene_id]
                if gene_symbol not in gene_features_dict:
                    continue
                gene_features = gene_features_dict[gene_symbol]

                molecule_id = row[0]
                samples_batch = np.append(samples_batch, np.asarray(drug_features + gene_features))
            samples_batch = samples_batch.reshape([num_genes, -1])

            # predict the batch
            predictions = model.predict(samples_batch)
            prediction_counter = 0
            for prediction in predictions:
                down_probability = prediction[0]
                if down_probability > 0.5:
                    gene_symbol = gene_id_dict[lm_gene_entrez_ids[prediction_counter % num_genes]]
                    message = gene_symbol + " " + str(down_probability) + " Found compound " + str(molecule_id) \
                              + " that downregulates " + gene_symbol + " " + str(down_probability)
                    if gene_symbol not in top10s:
                        top10s[gene_symbol] = Top10()
                        top10s[gene_symbol].add_item(down_probability, message)
                        print(message)
                    else:
                        if down_probability > top10s[gene_symbol].get_lowest_key():
                            top10s[gene_symbol].add_item(down_probability, message)
                            print(message)
                prediction_counter += 1
            drug_counter += 1
finally:
    en.notify("Predicting Done")