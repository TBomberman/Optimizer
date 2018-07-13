import datetime
import json
import numpy as np
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv
from sortedcontainers import SortedDict

gene_count_data_limit = 978
germans_up_genes = {} # ['7852', '3815']
germans_down_genes = ['6657'] # ['6657', '652']

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

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

# import os
# cwd = os.getcwd()
# print(cwd)
print(datetime.datetime.now(), "Loading drug and gene features")
drug_features_dict = get_feature_dict('data/smiles_rdkit_maccs.csv') #, use_int=True)
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)

# getting the gene ids
gene_id_dict = get_gene_id_dict()
# lm_gene_entrez_ids = list(gene_id_dict.keys())[:200]
lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)


print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)
repeat_X = {}
top10scores = Top10Float()

pert_scores_all_cell_lines = {}
flat_scores_all_cell_lines = {}

# For every experiment
print("Loading experiments")
for i in range(length-1, -1, -1): # go backwards, assuming later experiments have stronger perturbation
    printProgressBar(length - i, length, prefix='Load experiments progress')

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

    # parse the dosage unit and value
    dose_unit = experiment_data[5]
    if dose_unit != 'µM':
        continue
    dose_amt = float(experiment_data[4])
    if dose_amt == 0:
        if experiment_data[6] == '1 nM':
            dose_amt = 0.001
        else:
            print("Omitting 0 dose.\n")
            continue

    # parse the cell name
    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]

    if cell_name != 'VCAP':
        continue
    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]

    repeat_key = drug_id + "_" + cell_id
    if repeat_key in repeat_X and dose_amt <= repeat_X[repeat_key]:
        continue
    repeat_X[repeat_key] = dose_amt

    pert_score = 0.0
    pert_count = 0
    flat_score = 0.0
    flat_count = 0
    for gene_id in lm_gene_entrez_ids:
        pert = column[gene_id]
        if gene_id in germans_down_genes:
            pert_score += -pert
            pert_count += 1
        if gene_id in germans_up_genes:
            pert_score += pert
            pert_count += 1
        else:
            flat_score += abs(pert)
            flat_count += 1
    cell_drug_avg_pert_score = pert_score / pert_count
    cell_drug_avg_flat_score = flat_score / flat_count

    if cell_id not in pert_scores_all_cell_lines:
        pert_scores_all_cell_lines[cell_id] = {}
        flat_scores_all_cell_lines[cell_id] = {}

    if drug_id not in pert_scores_all_cell_lines[cell_id]:
        pert_scores_all_cell_lines[cell_id][drug_id] = []
        flat_scores_all_cell_lines[cell_id][drug_id] = []

    pert_scores_all_cell_lines[cell_id][drug_id].append(cell_drug_avg_pert_score)
    flat_scores_all_cell_lines[cell_id][drug_id].append(cell_drug_avg_flat_score)

scores = []
for drug_id in drug_features_dict.keys():
    cell_drug_pert_score_sum = 0.0
    cell_drug_flat_score_sum = 0.0
    cell_count = 0
    for cell in pert_scores_all_cell_lines.keys():
        if drug_id not in pert_scores_all_cell_lines[cell]:
            continue
        cell_drug_pert_score_sum += np.mean(pert_scores_all_cell_lines[cell][drug_id])
        cell_drug_flat_score_sum += np.mean(flat_scores_all_cell_lines[cell][drug_id])
        cell_count += 1

    if cell_count <= 0:
        continue

    total_score = cell_drug_pert_score_sum / cell_drug_flat_score_sum
    scores.append(total_score)
    if top10scores.current_size < top10scores.max_size or total_score > top10scores.get_lowest_key_prefix():
        message = "Compound " + str(drug_id) \
        + " has pert score " + "{:0.4f}".format(cell_drug_pert_score_sum/cell_count) \
        + " flat score " + "{:0.4f}".format(cell_drug_flat_score_sum/cell_count) \
        + " total score " + "{:0.4f}".format(total_score) \
        + " cell count " + str(cell_count)
        top10scores.add_item(total_score, message)
        print(datetime.datetime.now(), message)

def save_list(list, direction):
    prefix = "save_counts/"
    file = open(prefix + direction + '.txt', 'w')
    for item in list:
        file.write(str(item) + "\n")

save_list(scores, 'scores')