# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:37:16 2017

@author: mllamosa
"""

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
#import matplotlib.pyplot as plt
from data_loader import ensure_number, load_drug_single_gene_csv, load_descriptors, join_descriptors_label

import data_driven as dd
import mach_learn as ml
from utilities import all_stats,remove_constant_values,reduce_corr

import numpy as np

expression = load_drug_single_gene_csv('/data/datasets/mllamosa/WorkingData/Y_drug_id_one_expression.csv')
descriptors = load_descriptors('/data/datasets/mllamosa/WorkingData/X_all_descriptors.tab')
[data,labels] = join_descriptors_label(expression,descriptors)

# filter1 = remove_constant_values()
# ids_filter1 = filter1.apply(data)
# data1 = data[:,ids_filter1]
# #
# filter2 = reduce_corr()
# ids_filter2= filter2.apply(data1)
# data2 = data1[:,ids_filter2]

y = labels

train_fraction = 0.7
cut = int(len(labels)*train_fraction)
ids = np.arange(len(labels))
np.random.shuffle(ids)
train_id = ids[0:cut]
test_id = ids[cut:]

# turn labels into binary
#cutoff = np.mean(labels)
cutoff = -4
y = np.zeros((len(labels), 1)).astype(int)
pos_id = np.where(labels < cutoff)[0]
y[pos_id] = 1

partition = dd.Partition(trainset=train_id, testset=test_id)

data = np.nan_to_num(data)
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
data = np.nan_to_num(data)

model = dd.Model(data=np.hstack((data, y)),function=ml.RandF(parameters={'n_estimators':100, 'min_samples_split':50}), partition=partition,nfo=3)
#model = dd.Model(data=np.hstack((data, y)),function=ml.Mlr(), partition=partition,nfo=3)
# model = dd.Model(data=np.hstack((data,labels)),function=ml.Sksvm(parameters={'regularization':0.1,'sigma':0.1}),partition=partition)
model.training()
# model.crossvalidating()
model.testing()
model.performance()
model.summary()

# #lr = linear_model.LinearRegression()
#lr = RandomForestRegressor()
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
#predicted = cross_val_predict(lr, data, y, cv=3)
#
# fig, ax = plt.subplots()
# ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
