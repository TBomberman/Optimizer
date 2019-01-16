print("start script")
import csv
import datetime
import json
from pathlib import Path
import numpy as np
from L1000.gene_predictor import load_model
import os
import sys
import time
from multiprocessing import Pool
import multiprocessing as mp
from contextlib import closing
print("finished common imports")

from keras.models import model_from_json
print("finished keras import")

import helpers.email_notifier as en
print("finished my own imports")

# data_path = '/home/integra/projects/def-cherkaso/integra/ZINC_15_morgan_2048_2D/'
# save_path = '/home/integra/projects/def-cherkaso/integra/ZINC_15_morgan_2048_2D_scores/'
data_path = '/home/integra/Python/Optimizer/L1000/LDS-1191/data/'
save_path = '/home/integra/Python/Optimizer/L1000/LDS-1191/data/'
saved_model_path_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/"
# saved_model_path_prefix = '/home/integra/Data/screen_ar_models/'
up_model_file_prefix = "VCAP_NK_LM_AR_Up"
down_model_file_prefix = "VCAP_NK_LM_AR_Down"
print("got variables")


def load_csv(file):
    # load data
    expression = []
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel')
        for row in reader:
            expression.append(row)
    return expression


def get_feature_dict(file, delimiter=',', key_index=0, use_int=False):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        if use_int:
            my_dict = {}
            for row in reader:
                list = []
                for value in row[1:]:
                    list.append(int(value))
                my_dict[row[key_index]] = list
            return my_dict
        return dict((row[key_index], row[1:]) for row in reader)


# load model
def load_model(file_prefix):
    # load json and create model
    json_file = open(file_prefix + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_prefix + '.h5')
    print("Loaded model", file_prefix)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)


def get_gene_id_dict():
    lm_genes = json.load(open('/home/integra/Python/Optimizer/L1000/LDS-1191/data/lm_plus_ar_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict


def get_genes_features_list(up_gene_ids, down_gene_ids):
    gene_features_dict = get_feature_dict('/home/integra/Python/Optimizer/L1000/LDS-1191/data/lm_ar_gene_go_fingerprint.csv', use_int=True)
    gene_ids_by_var = load_csv('/home/integra/Python/Optimizer/L1000/LDS-1191/data/genes_by_var_lm_ar.csv')

    gene_ids_list = []
    for sublist in gene_ids_by_var:
        for item in sublist:
            gene_ids_list.append(item)

    up_gene_features_list = []
    down_gene_features_list = []
    for gene_id in gene_ids_list:
        gene_symbol = gene_id_dict[gene_id]
        if gene_symbol not in gene_features_dict:
            continue
        gene_features = gene_features_dict[gene_symbol]
        gene_features = np.asarray(gene_features)

        if gene_id in up_gene_ids:
            up_gene_features_list.append(gene_features)
        elif gene_id in down_gene_ids:
            down_gene_features_list.append(gene_features)

    return up_gene_features_list, down_gene_features_list


gene_id_dict = get_gene_id_dict()
ar_up_genes = ['7366', '7367', '10221']
ar_down_genes = ['367', '354', '3817', '7113', '991', '983', '890', '11065', '207']
up_gene_features_list, down_gene_features_list = get_genes_features_list(ar_up_genes, ar_down_genes)
n_genes = len(up_gene_features_list) + len(down_gene_features_list)
print("got more variables")


def get_drug_features(row):
    introw = []
    for item in row[1:]:
        introw.append(int(item))
    drug_features = np.zeros(2048)
    drug_features[introw] = 1
    return drug_features


def get_samples(drug_features, gene_features_list):
    samples_list = []
    for gene_features in gene_features_list:
        samples_list.append(np.concatenate([drug_features,gene_features]))
    return np.asarray(samples_list, dtype="float16")


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


def predict_batch(mol_id_batch, up_samples_batch, down_samples_batch, up_model, down_model):
    score_list = []
    n_samples = len(mol_id_batch)
    np_up_samples_batch = np.asarray(up_samples_batch)
    np_down_samples_batch = np.asarray(down_samples_batch)
    pert_score_batch = get_multi_mol_target_score(n_samples, np_up_samples_batch, np_down_samples_batch, up_model,
                                                  down_model, n_genes)
    for i in range(0, n_samples):
        # get one score per compound
        molecule_id_inner = mol_id_batch[i]
        pert_score = pert_score_batch[i]
        message = str(molecule_id_inner) + ", " + pert_score  # "{:0.6f}".format(pert_score)
        score_list.append(message)
        # print(datetime.datetime.now(), message)
    return score_list


def get_predictions(zinc_file_name, up_model, down_model):
    n_molecules_per_batch = 20000
    start_time = time.time()
    to_save = []

    with open(data_path + zinc_file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        drug_counter = 1
        mol_id_batch = []
        up_samples_batch = []
        down_samples_batch = []
        for row in reader:
            try:
                # if drug_counter % 100 == 0:
                #     print(drug_counter, time.time() - start_time)
                molecule_id = row[0]
                drug_features = get_drug_features(row)
                up_samples = get_samples(drug_features, up_gene_features_list)
                down_samples = get_samples(drug_features, down_gene_features_list)
                mol_id_batch.append(molecule_id)
                up_samples_batch.extend(up_samples)
                down_samples_batch.extend(down_samples)

                if drug_counter % n_molecules_per_batch == 0:
                    score_list = predict_batch(mol_id_batch, up_samples_batch, down_samples_batch, up_model, down_model)
                    to_save.extend(score_list)
                    mol_id_batch = []
                    up_samples_batch = []
                    down_samples_batch = []
            except:
                continue
            finally:
                drug_counter += 1
        if len(mol_id_batch) > 0:
            score_list = predict_batch(mol_id_batch, up_samples_batch, down_samples_batch, up_model, down_model)
            to_save.extend(score_list)
    with open(save_path + zinc_file_name + '.scores.csv', 'w') as f:
        for item in to_save:
            f.write("%s\n" % item)


def screen_for_ar_compounds(file):
    print("processing", file)
    start_time = time.time()
    try:
        up_model = load_model_from_file_prefix(saved_model_path_prefix + up_model_file_prefix)
        down_model = load_model_from_file_prefix(saved_model_path_prefix + down_model_file_prefix)
        get_predictions(file, up_model, down_model)
    finally:
        print(file, "processed", time.time() - start_time)
        en.notify("Predicting Done " + file)


def split_multi_process(n_sections=1, working_section=0):
    print("split_multi_process")
    files = os.listdir(data_path)
    existing_files = os.listdir(save_path)
    smi_files = []

    n_files = len(files)
    n_files_per_section = int(n_files/n_sections)
    start = working_section * n_files_per_section
    end = start + n_files_per_section

    for i in range(start, end):
        file = files[i]
        if file + '.scores.csv' not in existing_files:
            smi_files.append(file)

    try:
        with closing(Pool(mp.cpu_count())) as pool:
        # with closing(Pool(1)) as pool:
            pool.map(screen_for_ar_compounds, smi_files)
    finally:
        en.notify("Predicting Done All Files")


n_sections = int(sys.argv[0])
working_section = int(sys.argv[1])  # zero based
split_multi_process(n_sections, working_section)
