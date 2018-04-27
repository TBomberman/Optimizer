from L1000.gene_predictor import Gene_Predictor
import csv
import numpy as np
import helpers.email_notifier as en
import datetime
from sortedcontainers import SortedDict

def get_predictions(train_data, train_labels, gene_features_dict={}, lm_gene_entrez_ids=[], gene_id_dict={}):
    model = Gene_Predictor()
    model.train(train_data, train_labels)

    file_name = '/data/datasets/gwoo/tox21/ZincCompounds_WaitOK_maccs_full.csv'
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
                                del top10[top10.keys()[0]]
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