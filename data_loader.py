
import numpy as np
from mlp_optimizer import do_optimize
import csv

# local vars
cutoff = 0.5

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

def load_drug_single_gene_csv(file):
        #load data
        expression = []
        with open(file, "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel')
            for row in reader:
                expression.append(row)

        print('gene expressions loaded. rows:  ' + str(len(expression)))
        return np.array(expression)

def load_descriptors(file):
        descriptors = []
        with open(file, "r") as tab_file:
            reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
            descriptors = dict((rows[1],rows[2:]) for rows in reader)

        print('drug descriptors loaded. rows:  ' + str(len(descriptors)))
        return np.array(descriptors)


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
        return [data,labels]


