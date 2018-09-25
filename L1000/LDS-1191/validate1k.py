import numpy as np
import json
import datetime
from keras.utils import np_utils
from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

gene_count_data_limit = 978
cold_drugs_filename = '../cold_drugs.txt'
target_cell_name = 'A549'
nb_classes = 3
path_prefix = "saved_models/"
ends_model_file_prefix = "A549_Multi10"

# get entrez genes
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

def validate1k():
    experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
    drug_features_dict = get_feature_dict('data/smiles_rdkit_maccs.csv') #, use_int=True)
    cold_drugs_keys = load_csv(cold_drugs_filename)
    cell_name_to_id_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/Cell_Line_Metadata.txt', '\t', 2)
    gene_id_dict = get_gene_id_dict()
    gene_features_dict = get_feature_dict('data/gene_go_fingerprint_moreThan3.csv')#, use_int=True)

    print("Loading gene expressions from gctx")
    level_5_gctoo = load_gene_expression_data(
        "/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        lm_gene_entrez_ids)
    print("Gctx file in memory")

    length = len(level_5_gctoo.col_metadata_df.index)
    cell_X = {}
    cell_Y = {}
    cell_Y_gene_ids = {}
    cell_drugs_counts = {}
    repeat_X = {}
    bin = 10

    # For every experiment
    print("Loading experiments")
    for i in range(length-1, -1, -1): # go backwards, assuming later experiments have stronger perturbation
        printProgressBar(length - i, length, prefix='Load experiments progress')
        X = []
        Y = []

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
        dose_unit = experiment_data[5]
        if dose_unit != 'µM':  # standardize dose amounts
            continue
        dose_amt = float(experiment_data[4])
        if dose_amt < bin - 0.1 or dose_amt > bin + 0.1:  # only use the 5 mm bin
            continue

        drug_id = experiment_data[0]
        if drug_id not in drug_features_dict:
            continue
        drug_features = drug_features_dict[drug_id]

        if [drug_id] not in cold_drugs_keys: # <------ here's where you only get the 1000 kept away molecules
            continue

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]
        if cell_name != target_cell_name:
            continue

        if cell_name not in cell_name_to_id_dict:
            continue
        cell_id = cell_name_to_id_dict[cell_name][0]

        for gene_id in lm_gene_entrez_ids:
            gene_symbol = gene_id_dict[gene_id]

            if gene_symbol not in gene_features_dict:
                continue

            pert = column[gene_id].astype('float16')
            abspert = abs(pert)

            repeat_key = drug_id + "_" + cell_id + "_" + gene_id
            if repeat_key in repeat_X and abspert <= repeat_X[repeat_key]:
                continue

            if cell_id not in cell_X:
                cell_X[cell_id] = {}
                cell_Y[cell_id] = {}
                cell_drugs_counts[cell_id] = 0
                cell_Y_gene_ids[cell_id] = []

            repeat_X[repeat_key] = abspert

            cell_X[cell_id][repeat_key] = drug_features + gene_features_dict[gene_symbol]
            cell_Y[cell_id][repeat_key] = pert
            cell_Y_gene_ids[cell_id].append(gene_id)

    # use_global gene_specific_cutoffs:
    gene_cutoffs_down = {}
    gene_cutoffs_up = {}
    percentile_down = 5 # for downregulation, use 95 for upregulation
    percentile_up = 95
    prog_ctr = 0
    for gene_id in lm_gene_entrez_ids:
        row = level_5_gctoo.data_df.loc[gene_id, :].values
        prog_ctr += 1
        printProgressBar(prog_ctr, gene_count_data_limit, prefix='Storing percentile cutoffs')
        gene_cutoffs_down[gene_id] = np.percentile(row, percentile_down)
        gene_cutoffs_up[gene_id] = np.percentile(row, percentile_up)

    # mark the classes
    cell_id = cell_name_to_id_dict[target_cell_name][0]
    print(datetime.datetime.now(), "Converting dictionary values to np")
    npX = np.asarray(list(cell_X[cell_id].values()), dtype='float16')
    npY = np.asarray(list(cell_Y[cell_id].values()), dtype='float16')
    npY_gene_ids = np.asarray(cell_Y_gene_ids[cell_id])

    npY_class = np.zeros(len(npY), dtype=int)

    prog_ctr = 0
    combined_locations = []
    for gene_id in lm_gene_entrez_ids:  # this section is for gene specific class cutoffs
        prog_ctr += 1
        printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
        class_cut_off_down = gene_cutoffs_down[gene_id]
        class_cut_off_up = gene_cutoffs_up[gene_id]
        gene_locations = np.where(npY_gene_ids == gene_id)
        down_locations = np.where(npY <= class_cut_off_down)
        up_locations = np.where(npY >= class_cut_off_up)
        intersect = np.intersect1d(gene_locations, down_locations)
        gene_down_locations = intersect.tolist()
        combined_locations += gene_down_locations
        intersect = np.intersect1d(gene_locations, up_locations)
        gene_up_locations = intersect.tolist()
        combined_locations += gene_up_locations
        npY_class[gene_up_locations] = 2
        npY_class[gene_down_locations] = 1

    sample_size = len(npY_class)
    print('Positive samples', np.sum(npY_class))
    num_drugs = cell_drugs_counts[cell_id]
    print("Sample Size:", sample_size, "Drugs tested:", num_drugs / gene_count_data_limit)

    y_true = np_utils.to_categorical(npY_class, nb_classes)

    from pathlib import Path
    from L1000.gene_predictor import load_model
    from helpers.utilities import all_stats

    # load the model
    def load_model_from_file_prefix(model_file_prefix):
        model_file = Path(model_file_prefix + ".json")
        if not model_file.is_file():
            print(model_file.name + "File not found")
        return load_model(model_file_prefix)

    model = load_model_from_file_prefix(path_prefix + ends_model_file_prefix)

    y_pred = model.predict(npX)

    def print_stats(val_stats):
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
        print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.3f}'.format(val) for val in [val_stats[0]]])

    for class_index in range(0, nb_classes):
        print('class', class_index, 'stats')
        val_stats= all_stats(y_true[:, class_index], y_pred[:, class_index])
        print_stats(val_stats)

def validate_blind():
    blind_drugs_filename = 'data/blind_drugs.csv'
    blind_drugs = load_csv(blind_drugs_filename)
    strings = []
    for item in blind_drugs:
        strings.append(item[0])
    cold_ids = np.load("/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/5pBlindGapFixed/HT29_Multi_10b_10p_0g_cold_ids.npz")['arr_0']
    for id in cold_ids:
        if id in strings:
            print("uhoh", id)

def check_class_dist():
    file = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/10pBlindGapFixed/HT29_Multi_10b_10p_40g_npY_class.npz"
    test_classes = np.load(file)['arr_0']
    test_classes = np_utils.to_categorical(test_classes, nb_classes)
    print("0", np.sum(test_classes[:,0]))
    print("1", np.sum(test_classes[:, 1]))
    print("2", np.sum(test_classes[:, 2]))

def validate_landmark():
    gene_id_dict = get_gene_id_dict()
    for entry in gene_id_dict:
        print(entry)

validate_landmark()