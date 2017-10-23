#!/usr/bin/env python
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import sys
sys.path.append('C:/Users/mllamosa/Dropbox/repo/magpie')

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import regularizers
from keras import losses
import random
from random import sample
random.seed(1337)
from utilities import minmax, remove_constant_values, all_stats
import sys
from sklearn.metrics import mean_squared_error

cutoff = 0.5
batch_size = 200
nb_classes = 2
nb_epoch = 100#1000 cutoff 1 #3000 cutoff  2 and 5

dropout = 0.5
#Number of neurons dense
dense = 100

data_3 = np.loadtxt('/home/gwoo/Documents/Data/png_openeye/tox21_deepchem_InductiveDescriptors_qsar_label_NR-AR.csv', skiprows=1, delimiter=",", usecols = range(1,52))


data = data_3[:,0:52]
labels = data_3[:,-1]
print(labels)
y = np.zeros((len(labels),1)).astype(int)
pos_id = np.where(abs(labels)>cutoff)[0]
y[pos_id] = 1

print ("Number of positive class examples is ",len(pos_id))
'''Remove constant'''
max_X = np.amax(data[:,:-1])
min_X = np.amin(data[:,:-1])
data[:,:-1] = (data[:,:-1]-min_X)/(max_X-min_X)
data[np.where(data[:,0:-1]==np.NAN),0:-1] = 0

work_id = range(len(y))
train_partition = sample(work_id,int(0.7*len(work_id)))
test_partition = np.setdiff1d(work_id,train_partition)
val_partition = test_partition[0:int(len(test_partition)*0.5)]
test_partition = np.setdiff1d(test_partition,val_partition)

X_train = data[train_partition,:-1]
y_train = y[train_partition]

X_val = data[val_partition,:-1]
y_val = y[val_partition]

X_test = data[test_partition,:-1]
y_test = y[test_partition]

print("X train shape before",X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

print("X train shape after",X_train.shape)

#Normalize
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

# Y_train = y_train
# Y_test = y_test
# Y_val = y_val

  
model = Sequential()
#model.add(Dense(dense, input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l2(0.8),activity_regularizer=regularizers.l1(0.8)))
model.add(Dense(dense, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
model.add(Dropout(dropout))
#model.add(Dense(dense,kernel_regularizer=regularizers.l2(0.8),activity_regularizer=regularizers.l1(0.8)))
model.add(Dense(dense))
model.add(Activation('tanh'))
model.add(Dropout(dropout))
model.add(Dense(dense))
model.add(Activation('tanh'))
model.add(Dropout(dropout))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['binary_accuracy'])
# model.compile(loss=losses.mean_squared_error, optimizer='sgd',metrics=['mae', 'acc'])
#model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy',mean_squared_error])
#model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

y_score_train = model.predict_proba(X_train)
y_score_test = model.predict_proba(X_test)
y_score_val = model.predict_proba(X_val)

print('Test score:', score[0])
print('Test accuracy:', score[1])

train_stats = all_stats(Y_train[:,1],y_score_train[:,1])
test_stats = all_stats(Y_test[:,1],y_score_test[:,1],train_stats[-1])
val_stats = all_stats(Y_val[:,1],y_score_val[:,1],train_stats[-1])

#elif tag == "mlp":
# train_stats = all_stats(Y_train,y_score_train)
# test_stats = all_stats(Y_test,y_score_test,train_stats[-1])
# val_stats = all_stats(Y_val,y_score_val,train_stats[-1])
    
print('All stats train:',['{:6.2f}'.format(val) for val in train_stats])
print('All stats test:',['{:6.2f}'.format(val) for val in test_stats])
print('All stats val:',['{:6.2f}'.format(val) for val in val_stats])
#exit()
# np.savetxt('C:/Users/mllamosa.PROSTATECENTRE/Dropbox/2017/UBC_Prostate/Tox/deepchem/tox21_desc_cutoff_'+ str(cutoff) + '_val_10k_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:])    ,np.hstack((Y_val[:,0:1],y_score_val[:,0:1])),fmt="%s")
# np.savetxt('C:/Users/mllamosa.PROSTATECENTRE/Dropbox/2017/UBC_Prostate/Tox/deepchem/tox21_desc_cutoff_' + str(cutoff) + '_test_10k_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:])    ,    np.hstack((Y_test[:,0:1],y_score_test[:,0:1])),fmt="%s")
#
# print('C:/Users/mllamosa.PROSTATECENTRE/Dropbox/2017/UBC_Prostate/Tox/deepchem/tox21_desc_real_output_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_    ".join(sys.argv[1:]))
# fileout = open('C:/Users/mllamosa.PROSTATECENTRE/Dropbox/2017/UBC_Prostate/Tox/deepchem//tox21_desc_real_output_cutoff_'+ str(cutoff) +'_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:]),'w')
# fileout.write('All stats train:' + " ".join(['{:6.2f}'.format(val) for val in train_stats]) + "\n")
# fileout.write('All stats test:' + " ".join(['{:6.2f}'.format(val) for val in test_stats]) + "\n")
# fileout.write('All stats val:' + " ".join(['{:6.2f}'.format(val) for val in val_stats]) + "\n")
# fileout.close()
#
# if 'test' in sys.argv:
#
#     np.savetxt('/home/mllamosa/data/tox21_desc_val_10k_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:]),np.hstack((Y_val[:,0:1],y_score_val[:,1:])),fmt="%s")
#     np.savetxt('/home/mllamosa/data/tox21_desc_test_10k_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:]),np.hstack((Y_test[:,0:1],y_score_test[:,1:])),fmt="%s")
#     data_3_pred = np.loadtxt('/home/fer19x/data/data61/data/nandun/rcmat_10k.csv', skiprows=1, delimiter=",", usecols = range(0,362),dtype="str")
#     data_pred = data_3_pred[:,1:].astype(float)
#
#     Y_label = data_3_pred[:,0:1]
#     print("Shape of X pred",data_pred.shape)
#     data_pred[:,:-1] = (data_pred[:,:-1]-min_X)/(max_X-min_X)
#     X_pred = data_pred.reshape((data_pred.shape[0],1 ,img_rows,img_cols))
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     Y_pred = model.predict_proba(X_pred)
#     print(Y_pred)
#     np.savetxt('/home/fer19x/data/data61/data/dH_gap_pred_10k_img_' + str(img_rows) + "_" + str(img_cols)  + '_hyperparameters_'  + "_".join(sys.argv[1:]),np.hstack((Y_label,Y_pred[:,0:1])),fmt="%s")
#
#
