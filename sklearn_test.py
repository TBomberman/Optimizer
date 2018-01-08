# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:37:16 2017

@author: mllamosa
"""

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn import random_forest
import matplotlib.pyplot as plt
import numpy as np

lr = linear_model.LinearRegression()
lr = RandomForestRegressor()
boston = datasets.load_boston()
y = boston.target

data_3 = np.loadtxt(
    'C:/Users/mllamosa.PROSTATECENTRE/Dropbox/2017/UBC_Prostate/pcba/pcba_deepchem_rdkit_morgan_pcba.csv', skiprows=1,
    delimiter=",", usecols=range(1, 2109))
df[].as_matrix()

data = data_3[:, 0:2048]
labels = data_3[:, 2048:2049]
nb_classes = labels.shape[1]
y = labels

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, data, y, cv=3)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(data_3[:, 2056], data_3[:, 2055], edgecolors=(0, 0, 0))
plt.show()