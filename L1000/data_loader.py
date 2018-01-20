from cmapPy.pandasGEXpress import parse
import numpy as np
import csv

# local vars
cutoff = 0.5

def ensure_number(data):
    return np.nan_to_num(data)

def load_features_csv(file):
        #load data
        expression = []
        with open(file, "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel')
            for row in reader:
                expression.append(row)

        print('gene expressions loaded. rows:  ' + str(len(expression)))
        return expression

def load_descriptors(file):
        descriptors = []
        with open(file, "r") as tab_file:
            reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
            descriptors = dict((rows[1],rows[2:]) for rows in reader)

        print('drug descriptors loaded. rows:  ' + str(len(descriptors)))
        return descriptors

def join_descriptors_label(expression,descriptors):
        unique_drugs = []
        # data set up
        data = []
        for row in expression:
            data.append(descriptors[row[0]])
            if row[0] not in unique_drugs:
                unique_drugs.append(row[0])
        data = np.array(data).astype(np.float32)

        labels = []
        for row in expression:
            labels.append(row[1:3])

        labels = np.array(labels).astype(np.float32)
        print('data size ' + str(len(data)) + ' labels size ' + str(len(labels)))
        return data,labels

def get_feature_dict(file, delimiter=',', key_index=0):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        return dict((row[key_index], row[1:]) for row in reader)

def load_gene_expression_data(lm_gene_entrez_ids):
    return parse(
        "/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        col_meta_only=False, row_meta_only=False, rid=lm_gene_entrez_ids)