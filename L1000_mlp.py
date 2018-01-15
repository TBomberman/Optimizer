import numpy as np

from L1000.data_loader import ensure_number, load_drug_single_gene_csv, load_descriptors, join_descriptors_label
from mlp_optimizer import do_optimize

# local vars
cutoff = 0.5

expression = load_drug_single_gene_csv('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/Y_drug_id_one_expression.csv')
descriptors = load_descriptors('/data/datasets/gwoo/L1000/LDS-1191/WorkingData/X_all_descriptors.tab')
[data,labels] = join_descriptors_label(expression,descriptors)

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

