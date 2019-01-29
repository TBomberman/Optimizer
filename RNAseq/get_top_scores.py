import os
import csv
import numpy as np
import sklearn.metrics as metrics
from helpers.plot_roc import plot_roc

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

def compare_predictions_with_nate():
    # do this on the gpu cluster
    from L1000.data_loader import load_csv
    from RNAseq.get_predictions import predict_nathans
    from helpers.utilities import all_stats

    # get the predictions np array
    up_predictions, down_predictions, drugs, genes = predict_nathans()

    # get the Nate's data into np array
    csv_file = load_csv('data/DESeq2results__3reps_get_classes.csv')
    old_to_new_symbol = {
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
    nates_data = {}
    for line in csv_file[1:]:
        drug = line[0]
        gene = line[1]
        actives = line[2:12]
        if drug not in nates_data:
            nates_data[drug] = {}
        if gene not in nates_data[drug]:
            nates_data[drug][gene] = actives
    up1_true = []
    up5_true = []
    up10_true = []
    up15_true = []
    up20_true = []
    down1_true = []
    down5_true = []
    down10_true = []
    down15_true = []
    down20_true = []

    for drug in drugs:
        for gene in genes:
            if gene in old_to_new_symbol:
                gene = old_to_new_symbol[gene]
            if gene not in nates_data[drug]:
                print('nate missing gene', gene)
                continue
            up1_true.append(nates_data[drug][gene][0])
            up5_true.append(nates_data[drug][gene][1])
            up10_true.append(nates_data[drug][gene][2])
            up15_true.append(nates_data[drug][gene][3])
            up20_true.append(nates_data[drug][gene][4])
            down1_true.append(nates_data[drug][gene][5])
            down5_true.append(nates_data[drug][gene][6])
            down10_true.append(nates_data[drug][gene][7])
            down15_true.append(nates_data[drug][gene][8])
            down20_true.append(nates_data[drug][gene][9])

    def print_stats(y_true, padj, dir, predictions, cutoff=None):
        val_stats = all_stats(np.asarray(y_true, dtype='float32'), predictions[:, 1], cutoff)
        label = dir + "regulation padj " + str(padj)
        print(label)
        print_stats_inner(val_stats)
        print_acc(label, np.asarray(y_true, dtype='float32'), predictions)

    print_stats(up1_true, 0.01, "up", up_predictions, 0.561)
    print_stats(up5_true, 0.05, "up", up_predictions, 0.561)
    print_stats(up10_true, 0.1, "up", up_predictions, 0.561)
    print_stats(up15_true, 0.15, "up", up_predictions, 0.561)
    print_stats(up20_true, 0.2, "up", up_predictions, 0.561)
    plot_roc(np.asarray(up5_true, dtype='float32'), up_predictions[:, 1])
    # plot_roc(np.asarray(up15_true, dtype='float32'), up_predictions[:, 1])
    # plot_roc(np.asarray(up20_true, dtype='float32'), up_predictions[:, 1])

    # print("down support", sum(np.asarray(down1_true, dtype='float32')))
    # print("down support", sum(np.asarray(down5_true, dtype='float32')))
    # print("down support", sum(np.asarray(down10_true, dtype='float32')))
    # print("down support", sum(np.asarray(down15_true, dtype='float32')))
    # print("down support", sum(np.asarray(down20_true, dtype='float32')))
    print_stats(down1_true, 0.01, "down", down_predictions, 0.648)
    print_stats(down5_true, 0.05, "down", down_predictions, 0.648)
    print_stats(down10_true, 0.10, "down", down_predictions, 0.648)
    print_stats(down15_true, 0.15, "down", down_predictions, 0.648)
    print_stats(down20_true, 0.2, "down", down_predictions, 0.648)
    plot_roc(np.asarray(down5_true, dtype='float32'), down_predictions[:, 1])
    # plot_roc(np.asarray(down10_true, dtype='float32'), down_predictions[:, 1])
    # plot_roc(np.asarray(down15_true, dtype='float32'), down_predictions[:, 1])

    # compare using Mike's api




# smiless2sdf()
# testBiomart()
compare_predictions_with_nate()