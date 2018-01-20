from helpers import data_driven as dd, mach_learn as ml
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import History
from random import sample
from helpers.utilities import all_stats
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def do_optimize(nb_classes, data, labels, data_test=None, labels_test=None):
    if data_test is None:
        # ids
        work_ids = range(len(labels))
        train_ids = sample(work_ids, int(0.7 * len(work_ids)))
        test_ids = np.setdiff1d(work_ids, train_ids)
        val_ids = test_ids[0:int(len(test_ids) * 0.5)]
        test_ids = np.setdiff1d(test_ids, val_ids)

        # X data
        X_train = data[train_ids, :]
        X_test = data[test_ids, :]
        X_val = data[val_ids, :]

        # Y data
        y_train = labels[train_ids]
        y_test = labels[test_ids]
        y_val = labels[val_ids]
    else:
        # ids
        test_ids = range(len(labels_test))
        val_ids = test_ids[0:int(len(test_ids) * 0.5)]
        test_ids = np.setdiff1d(test_ids, val_ids)

        # X data
        X_train = data
        X_test = data_test[test_ids, :]
        X_val = data_test[val_ids, :]

        # Y data
        y_train = labels
        y_test = labels_test[test_ids]
        y_val = labels_test[val_ids]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train
    Y_test = y_test
    Y_val = y_val

    model = RandomForestRegressor(verbose=1)

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)
    tr_error = np.mean(y_pred != Y_train)
    print("Random Forest (sklearn) train error: %.3f" % tr_error)

    y_pred = model.predict(X_test)
    te_error = np.mean(y_pred != Y_test)
    print("Random Forest (sklearn) test error: %.3f" % te_error)

    y_pred = model.predict(X_val)
    v_error = np.mean(y_pred != Y_val)
    print("Random Forest (sklearn) validation error: %.3f" % v_error)


