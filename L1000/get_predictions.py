import csv
import numpy as np
import helpers.email_notifier as en
import datetime
from sortedcontainers import SortedDict
from pathlib import Path
from L1000.gene_predictor import load_model
from L1000.data_loader import get_feature_dict, load_csv
import json

model_file_prefix = "100PC3"
gene_count_data_limit = 100

# load model
model = object
model_file = Path(model_file_prefix + ".json")
if not model_file.is_file():
    print(model_file + "File not found")
model = load_model(model_file_prefix)

# load gene fingerprints to test
gene_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/gene_go_fingerprint.csv', use_int=True)
lm_gene_entrez_ids_list = load_csv('genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)
def get_gene_id_dict():
    lm_genes = json.load(open('landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict
gene_id_dict = get_gene_id_dict()

file_name = '/data/datasets/gwoo/tox21/ZincCompounds_InStock_maccs.tab'
top10 = SortedDict()
top10_full = False
top10_counter = 0

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

            for gene_id in lm_gene_entrez_ids:
                gene_symbol = gene_id_dict[gene_id]
                if gene_symbol not in gene_features_dict:
                    continue
                gene_features = gene_features_dict[gene_symbol]

                molecule_id = row[0]
                data = np.asarray([drug_features + gene_features])
                prediction = model.predict(data)
                down_probability = prediction[0][0]
                if down_probability > 0.5:
                    message = "Found compound " + str(molecule_id) + " that downregulates " + gene_symbol + " " + str(down_probability)
                    if top10_full:
                        if down_probability > top10.keys()[0]:
                            top10.popitem(False)
                            top10[down_probability] = message
                            print(message)
                    else:
                        top10[down_probability] = message
                        print(message)
                        top10_counter += 1
                        if top10_counter >= 10:
                            top10_full = True

            drug_counter += 1
finally:
    print(top10)
    en.notify("Predicting Done")