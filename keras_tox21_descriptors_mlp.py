import numpy as np
from mlp_optimizer import do_optimize

# local vars
cutoff = 0.5

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

#load data
dataset = np.loadtxt('/data/datasets/gwoo/png_openeye/tox21_deepchem_InductiveDescriptors_qsar_label_NR-AR-LBD.csv',
                     skiprows=1, delimiter=",", usecols = range(1,52))

# data set up
data = dataset[:, :-1]
labels = dataset[:, -1]

# put data between 0-1
max_X = np.amax(data)
min_X = np.amin(data)
data = (data - min_X) / (max_X - min_X)  # data bewteen 0-1, does this take in to account negatives?

# turn labels into binary
y = np.zeros((len(labels), 1)).astype(int)
pos_id = np.where(abs(labels) > cutoff)[0]
y[pos_id] = 1

# validate every data is a number
ensure_number(data)
ensure_number(y)

do_optimize(2, data, y)

