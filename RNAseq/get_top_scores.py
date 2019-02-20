import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import numpy as np
import sklearn.metrics as metrics
from helpers.plot_roc import plot_roc, plot_precision_recall, plot_roc_multi
from scipy.stats import pearsonr, spearmanr
from L1000.data_loader import load_csv
from L1000.data_loader import get_feature_dict
from L1000.gene_predictor import load_model
from pathlib import Path
from helpers.utilities import all_stats

def compare_graham_cedar():
    scores_path = 'H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores'
    files = os.listdir(scores_path)


def testBiomart():
    from pybiomart import Dataset
    import json
    import math

    # get a dict from biomart of entrez and ensembl and name
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene'])
    biomart_dict = {}
    for index, row in df.iterrows():
        if math.isnan(row[2]):
            continue
        entrez = int(row[2])
        if str(entrez) in biomart_dict:
            biomart_dict[str(entrez)].append(row[0] + " " + row[1])
        else:
            biomart_dict[str(entrez)] = [row[0] + " " + row[1]]

    # get dict of nate's gene name and ensemls
    nate_dict = {}
    with open('data/GeneNameConversion_index.txt', "r") as csv_file:
        nate_reader = csv.reader(csv_file, dialect='excel', delimiter='\t')
        for row in nate_reader:
            nate_dict[row[2]] = row[1]

    # get list of landmark genes get the name and entrez
    # for each lm gene get the gene name,
    #   then print the biomart's name and ensembl from the entrez, nate's name and ensemble from the gene name
    lm_genes = json.load(open('../L1000/LDS-1191/data/lm_plus_ar_genes.json'))
    for lm_gene in lm_genes:
        lm_entrez = lm_gene['entrez_id']
        lm_symbol = lm_gene['gene_symbol']
        if lm_symbol not in nate_dict:
            print("nate missing lm_symbol", lm_symbol)
            continue
        nate_ens = nate_dict[lm_symbol]
        if lm_entrez not in biomart_dict:
            print("biomart missing entrez", lm_entrez)
            continue
        bio_list = biomart_dict[lm_entrez]
        symbol_found = False
        entrez_found = False
        for item in bio_list:
            bio_ens, bio_symbol = item.split()
            if lm_symbol == bio_symbol:
                symbol_found = True
            if bio_ens == nate_ens:
                entrez_found = True
        if not symbol_found:
            print("lm_symbol not found in biomart's list for ", lm_entrez)
        if not entrez_found:
            print("nate_ens not found in biomart's list for ", lm_symbol)


def print_stats_inner(val_stats):
    print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
    print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])


def print_acc(text, Y_train, y_pred_train):
    y_pred = np.argmax(y_pred_train, axis=1)
    y_true = Y_train
    target_names = [0, 1]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accs = cm.diagonal()
    print(text, "Accuracy class 0", accs[0])  # number of actual 0's predicted correctly
    print(text, "Accuracy class 1", accs[1])  # number of actual 1's predicted correctly

    report = metrics.classification_report(y_true, y_pred)
    print("Report", report)

def get_true_from_padj(drugs, genes, old_to_new_symbol, nates_data, significance_level):
    up_true_float = []
    down_true_float = []
    up_true_int = []
    down_true_int = []

    for drug in drugs:
        for gene in genes:
            if gene in old_to_new_symbol:
                gene = old_to_new_symbol[gene]
            if gene not in nates_data[drug]:
                print('nate missing gene', gene)
                continue
            padj = float(nates_data[drug][gene][1])
            log2change = float(nates_data[drug][gene][0])
            up_value = 0
            down_value = 0
            if log2change >= 0:
                if padj <= significance_level:
                    up_value = 1
                up_true_float.append(-padj)
                down_true_float.append(-1)
                up_true_int.append(up_value)
                down_true_int.append(0)
            else:
                if padj <= significance_level:
                    down_value = 1
                up_true_float.append(-1)
                down_true_float.append(-padj)
                up_true_int.append(0)
                down_true_int.append(down_value)
    return up_true_float, down_true_float, up_true_int, down_true_int


def get_true_from_log2change(drugs, genes, old_to_new_symbol, nates_data, significance_level):
    up_true_float = []
    down_true_float = []
    up_true_int = []
    down_true_int = []

    all_log_2_change = []
    for drug in drugs:
        for gene in genes:
            if gene in old_to_new_symbol:
                gene = old_to_new_symbol[gene]
            if gene not in nates_data[drug]:
                print('nate missing gene', gene)
                continue
            log2change = float(nates_data[drug][gene][0])
            all_log_2_change.append(log2change)
    down_threshold = np.percentile(all_log_2_change, 100 * significance_level)
    up_threshold = np.percentile(all_log_2_change, 100 * (1 - significance_level))

    for drug in drugs:  # it must be in this order
        for gene in genes:  # it must be in this order
            if gene in old_to_new_symbol:
                gene = old_to_new_symbol[gene]
            if gene not in nates_data[drug]:
                print('nate missing gene', gene)
                continue
            padj = float(nates_data[drug][gene][2])
            log2change = float(nates_data[drug][gene][0])
            up_value = 0
            down_value = 0
            if log2change >= 0:
                if log2change >= up_threshold:
                    up_value = 1
                up_true_float.append(log2change)
                down_true_float.append(-log2change)
                up_true_int.append(up_value)
                down_true_int.append(0)
            else:
                if log2change <= down_threshold:
                    down_value = 1
                up_true_float.append(log2change)
                down_true_float.append(-log2change)
                up_true_int.append(0)
                down_true_int.append(down_value)
    return up_true_float, down_true_float, up_true_int, down_true_int

lincs_to_rnaseq_gene = {
        'PAPD7': 'TENT4A',
        'HDGFRP3': 'HDGFL3',
        'TMEM2': 'CEMIP2',
        'TMEM5': 'RXYLT1',
        'SQRDL': 'SQOR',
        'KIAA0907': 'KHDC4',
        'IKBKAP': 'ELP1',
        'TMEM110': 'STIMATE',
        'NARFL': 'CIAO3',
        'HN1L': 'JPT2'
    }

rnaseq_to_lincs= {
        'TENT4A': 'PAPD7',
        'HDGFL3': 'HDGFRP3',
        'CEMIP2': 'TMEM2',
        'RXYLT1': 'TMEM5',
        'SQOR': 'SQRDL',
        'KHDC4': 'KIAA0907',
        'ELP1': 'IKBKAP',
        'STIMATE': 'TMEM110',
        'CIAO3': 'NARFL',
        'JPT2': 'HN1L'
    }


def print_stats(y_true, param, dir, predictions, cutoff=None):
    val_stats = all_stats(np.asarray(y_true, dtype='float32'), predictions[:, 1], cutoff)
    label = dir + "regulation " + str(param)
    print(label)
    print_stats_inner(val_stats)
    print_acc(label, np.asarray(y_true, dtype='float32'), predictions)


def compare_predictions_with_nate():
    # do this on the gpu cluster
    from RNAseq.get_predictions import predict_nathans

    # get the predictions np array
    up_predictions, down_predictions, drugs, genes = predict_nathans()  # ordered by drugs then genes

    # get the Nate's data into np array
    csv_file = load_csv('data/DESeq2results__3reps.csv')
    nates_data = {}
    for line in csv_file[1:]:
        drug = line[0]
        gene = line[1]
        padj = line[2:5]
        if drug not in nates_data:
            nates_data[drug] = {}
        if gene not in nates_data[drug]:
            nates_data[drug][gene] = padj

    significance_levels = [0.05]
    up_max_cutoff = 0.364
    down_max_cutoff = 0.606

    for significance_level in significance_levels:
        print("significance level", significance_level)

        up_true_float, down_true_float, up_true_int, down_true_int = \
            get_true_from_padj(drugs, genes, lincs_to_rnaseq_gene, nates_data, significance_level)

        print_stats(up_true_int, significance_level, "up", up_predictions, up_max_cutoff)
        int_pred = np.zeros(len(up_predictions[:, 1]))
        positive = np.where(up_predictions[:, 1] >= up_max_cutoff)
        int_pred[positive] = 1
        print("MCC value", metrics.matthews_corrcoef(up_true_int, int_pred))
        print("Pearson class", pearsonr(up_true_int, up_predictions[:, 1]))
        print("Pearson float", pearsonr(up_true_float, up_predictions[:, 1]))
        print("Spearman", spearmanr(up_true_int, up_predictions[:, 1]))
        y_true_list = []
        y_true_list.append(np.asarray(up_true_int, dtype='float32'))
        y_pred_list = []
        y_pred_list.append(up_predictions[:, 1])
        legend = []
        legend.append("Upregulation")
        # plot_precision_recall(np.asarray(up_true_int, dtype='float32'), up_predictions[:, 1],
        #                       title='PR and ROC Upregulation')

        print_stats(down_true_int, significance_level, "down", down_predictions, down_max_cutoff)
        int_pred = np.zeros(len(down_predictions[:, 1]))
        positive = np.where(down_predictions[:, 1] >= down_max_cutoff)
        int_pred[positive] = 1

        print("MCC value", metrics.matthews_corrcoef(down_true_int, int_pred))
        print("Pearson class", pearsonr(down_true_int, down_predictions[:, 1]))
        print("Pearson float", pearsonr(down_true_float, down_predictions[:, 1]))
        print("Spearman", spearmanr(down_true_float, down_predictions[:, 1]))
        # plot_precision_recall(np.asarray(down_true_int, dtype='float32'), down_predictions[:, 1],
        #                       title='PR and ROC Downregulation')
        y_true_list.append(np.asarray(down_true_int, dtype='float32'))
        y_pred_list.append(down_predictions[:, 1])
        legend.append("Downregulation")
        plot_roc_multi(y_true_list, y_pred_list, legend, 'ROC: RNAseq Predictions (24h)')

def compare_lm_files():
    import json
    genes1484 = json.load(open('../L1000/LDS-1484/data/landmark_genes.json'))
    genes1191 = json.load(open('../L1000/LDS-1191/data/landmark_genes.json'))
    genes1484dict = {}
    for gene in genes1484:
        genes1484dict[gene['entrez_id']] = 1
    genes1191dict = {}
    for gene in genes1191:
        genes1191dict[gene['entrez_id']] = 1

    count = 0
    for key in genes1484dict:
        if key not in genes1191dict:
            count += 1
            print(count, key, "not in 1191")
    print("done commparison")


# load model
def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)


def compare_predictions_with_nate_only_good_pvalues():
    def display_stats(true_class, true_float, significance_level, direction, predictions, model_max_cutoff,
                      l2c_threshold):
        param = 'sig level ' + str(significance_level) + ' l2c ' + str(l2c_threshold)
        print_stats(true_class, param, direction, predictions, model_max_cutoff)
        pred_class = np.zeros(len(predictions[:, 1]))
        positive = np.where(predictions[:, 1] >= model_max_cutoff)
        pred_class[positive] = 1
        print("MCC value", metrics.matthews_corrcoef(true_class, pred_class))
        print("Pearson class", pearsonr(true_class, predictions[:, 1]))
        print("Pearson float", pearsonr(true_float, predictions[:, 1]))
        print("Spearman", spearmanr(true_float, predictions[:, 1]))
        # plot_roc(np.asarray(true_class, dtype='float32'), predictions[:, 1],
        #          title='ROC Predicting Active ' + direction + 'regulations')
        plot_precision_recall(np.asarray(true_class, dtype='float32'), predictions[:, 1],
                 title='PR and ROC ' + direction + 'regulation ' + param)

    # get drugs and gene list
    gene_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/gene_go_fingerprint_moreThan3.csv')
    drug_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/nathans_morgan_2048_nk.csv')

    # load nates data
    csv_file = load_csv('data/DESeq2results__3reps.csv')

    # for each drug and gene save the ones that are good pvalues and it's class in up or down scenario and float
    for significance_level in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        l2c_thresholds = [0] #, 0.2, 0.4, 0.6, 0.8, 1]
        for l2c_threshold in l2c_thresholds:
            to_predict_dict = {}
            for line in csv_file[1:]:
                drug = line[0]
                gene = line[1]
                l2c = float(line[2])
                padj = float(line[3])
                pval = float(line[4])
                if drug in ['VPC220010', 'VPC13789']:  # can't make this public anymore
                    continue
                # if pval > significance_level:
                #     continue
                if gene in rnaseq_to_lincs:
                    gene = rnaseq_to_lincs[gene]
                if gene not in gene_features_dict:
                    continue
                if drug not in to_predict_dict:
                    to_predict_dict[drug] = {}
                up_class = 0
                down_class = 0

                if pval <= significance_level:
                    if l2c >= 0:
                        up_class = 1
                else:
                    if l2c < 0:
                        down_class = 1

                # if l2c >= l2c_threshold:
                #     up_class = 1
                # if l2c <= -l2c_threshold:
                #     down_class = 1
                to_predict_dict[drug][gene] = [l2c, -l2c, up_class, down_class]

            # get predictions
            model_input = []
            descriptions = []
            for drug in to_predict_dict:
                for gene in to_predict_dict[drug]:
                    model_input.append(drug_features_dict[drug] + gene_features_dict[gene])
                    descriptions.append(drug + ", " + gene)
            model_input = np.asarray(model_input, dtype=np.float16)
            up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1484/saved_models/closeToX10/2019-02-12 15:37:16.049247_LNCAP_Up_10b_5p_24h_repeat8"
            down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1484/saved_models/closeToX10/2019-02-12 15:37:29.049509_LNCAP_Down_10b_5p_24h_repeat9"
            up_model = load_model_from_file_prefix(up_model_filename_prefix)
            down_model = load_model_from_file_prefix(down_model_filename_prefix)
            up_y_pred = up_model.predict(model_input)
            down_y_pred = down_model.predict(model_input)

            up_y_true_class = []
            up_y_true_float = []
            down_y_true_class = []
            down_y_true_float = []
            for drug in to_predict_dict:
                for gene in to_predict_dict[drug]:
                    true_stuff = to_predict_dict[drug][gene]
                    up_y_true_class.append(true_stuff[2])
                    up_y_true_float.append(true_stuff[0])
                    down_y_true_class.append(true_stuff[3])
                    down_y_true_float.append(true_stuff[1])

            # compare
            display_stats(up_y_true_class, up_y_true_float, significance_level, "Up", up_y_pred, 0.485, l2c_threshold)
            display_stats(down_y_true_class, down_y_true_float, significance_level, "Down", down_y_pred, 0.581, l2c_threshold)

# smiless2sdf()
# testBiomart()
compare_predictions_with_nate()
# compare_lm_files()
# compare_predictions_with_nate_only_good_pvalues()