from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar
import json
from random_forest import do_optimize
import numpy as np
import gc

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_gene_id_dict():
    lm_genes = json.load(open('landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

# get the dictionaries
# get the expressions
print("Loading drug and gene features")
drug_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/smiles_rdkit_maccs.csv', use_int=True)
gene_features_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/gene_go_fingerprint.csv', use_int=True)
# info to separate by data by cell lines, drug + gene tests may not be equally spread out across cell lines
cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
# info to remove any dosages that are not 'µM'. Want to standardize the dosages.
experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
gene_id_dict = get_gene_id_dict()
lm_gene_entrez_ids = list(gene_id_dict.keys())[:50]
print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)

length = len(level_5_gctoo.col_metadata_df.index)

cell_X = {}
cell_Y = {}
cell_drugs = {}
repeat_X = {}

# For every experiment
print("Loading experiments")
one_percent = int(length/50)
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
    time = col_name[start + 1:end]
    if time != "24H": # column counts: 6h 95219, 24h 109287, 48h 58, 144h 1
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
    dose_unit = experiment_data[5]
    if dose_unit != 'µM': # standardize dose amounts
        # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
        continue
    dose_amt = float(experiment_data[4])

    # parse the cell name
    start = find_nth(col_name, "_", 1)
    end = find_nth(col_name, "_", 2)
    cell_name = col_name[start + 1:end]
    # cell_name = "A375"  # this line will combine all cell lines into one
    if cell_name not in cell_name_to_id_dict:
        continue
    cell_id = cell_name_to_id_dict[cell_name][0]

    for gene_id in lm_gene_entrez_ids:
        gene_symbol = gene_id_dict[gene_id]

        if gene_symbol not in gene_features_dict:
            continue

        if cell_id not in cell_X:
            cell_X[cell_id] = []
            cell_Y[cell_id] = []
            cell_drugs[cell_id] = []

        # repeat_key = drug_id + cell_id + gene_id + dose_amt
        repeat_key = drug_id + cell_id + gene_id
        if repeat_key in repeat_X: # duplicated experiments with different training perturbation will confuse the model
            continue
        repeat_X[repeat_key] = None

        cell_X[cell_id].append([dose_amt] + drug_features + gene_features_dict[gene_symbol])
        # cell_X[cell_id].append([dose_amt] + drug_features)
        cell_Y[cell_id].append(column[gene_id])
        cell_drugs[cell_id].append(drug_id)

gc.collect()
cell_line_counter = 1
print("Printing cell data.\n")
for cell_name in cell_name_to_id_dict:
    # print("Looking at", cell_name)
    cell_id = cell_name_to_id_dict[cell_name][0]
    if cell_id not in cell_X:
        # print("Skipping", cell_name, ". No cell line data.\n")
        continue

    npX = np.asarray(cell_X[cell_id])
    npY = np.asarray(cell_Y[cell_id])

    sample_size = len(npX)

    if sample_size < 300: # smaller sizes was giving y values of only one class
        # print("Skipping", cell_name, ". Sample size", sample_size, "is too small.\n")
        continue

    for i in range(0, 1): # to help iterate through classification thresholds
        percentile = 95 + i
        class_cut_off = np.percentile(npY, percentile)

        npY_class = np.zeros(len(npY), dtype=int)
        npY_class[np.where(npY > class_cut_off)] = 1

        print("Evaluating cell line", cell_line_counter, cell_name, "class cutoff", class_cut_off, "(Percentile:", percentile, ")")
        num_drugs = len(set(cell_drugs[cell_id]))
        print("Sample Size:", sample_size, "Drugs tested:", num_drugs)
        do_optimize(0, npX, npY_class)
        cell_line_counter += 1
        print()
