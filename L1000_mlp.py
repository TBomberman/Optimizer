
import numpy as np
from mlp_optimizer import do_optimize
import csv

# local vars
cutoff = 0.5

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

#load data
expression = []
with open('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/Y_drug_id_one_expression.csv', "r") as csv_file:
    reader = csv.reader(csv_file, dialect='excel')
    for row in reader:
        expression.append(row)

print('gene expressions loaded. rows:  ' + str(len(expression)))

descriptors = []
with open('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/X_all_descriptors.tab', "r") as tab_file:
    reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
    descriptors = dict((rows[1],rows[2:]) for rows in reader)

print('drug descriptors loaded. rows:  ' + str(len(descriptors)))

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

# put data between 0-1
max_X = np.nanmax(data)
min_X = np.nanmin(data)
data = (data - min_X) / (max_X - min_X)  # data bewteen 0-1


# put labels between 0-1
max_Y = np.nanmax(data)
min_Y = np.nanmin(data)
labels = (labels - min_Y) / (max_Y - min_Y)  # data bewteen 0-1

# # turn labels into binary
# y = np.zeros((len(labels), 1)).astype(int)
# pos_id = np.where(abs(labels) > cutoff)[0]
# y[pos_id] = 1

# validate every data is a number
ensure_number(data)
ensure_number(labels)

print('training data')
do_optimize(None, data, labels)

