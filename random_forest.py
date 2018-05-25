import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from random import sample
from helpers.utilities import scatter2D_plot
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from helpers.plot_roc import plot_roc
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

train_percentage = 0.7

def do_optimize(nb_classes, data, labels):
    n = len(labels)
    train_size = int(train_percentage * n)
    print("Train size:", train_size)
    test_size = int((1 - train_percentage) * n)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
    X_train = X_train.astype('float16')
    X_test = X_test.astype('float16')
    Y_train = y_train
    Y_test = y_test

    sum_test = np.sum(Y_test)
    sum_train = np.sum(Y_train)
    show_roc = False

    if sum_train > 0 and sum_train < len(Y_train) and sum_test > 0 and sum_test < len(Y_test):

        for hyperparam in range(1,20):
            print("hyperparam", hyperparam)
            # model = RandomForestRegressor(n_estimators=100, min_samples_split=50, n_jobs=-1)
            model = RandomForestClassifier(n_estimators=20, max_depth=2**hyperparam)

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_train)
            # uncomment below for RandomForestClassifier
            # tr_error = np.mean((y_pred - Y_train)**2)
            # scatter2D_plot(y_train, y_pred, "train", "rf")
            tr_auc = roc_auc_score(Y_train, y_pred)

            y_pred = model.predict(X_test)
            # uncomment below for RandomForestClassifier
            # te_error = np.mean((y_pred - Y_test)**2)
            # scatter2D_plot(Y_test, y_pred, "test", "rf")
            te_auc = roc_auc_score(Y_test, y_pred)

            # avg_test = (te_error + v_error)/2
            # avg_auc = (te_auc + v_auc) / 2
            # print("Random Forest (sklearn) error train: %.3f test: %.3f avg test: %.3f" % (tr_error, te_error, avg_test))
            # print("Random Forest (sklearn) AUC train: %.3f test: %.3f val: %.3f avg test: %.3f" % (
            #     tr_auc, te_auc, v_auc, avg_auc))
            accuracy = 1 - np.mean(y_pred != y_test)

            print("Random Forest (sklearn) AUC train: %.3f test: %.3f accuracy: %.3f" % (tr_auc, te_auc, accuracy))

            if show_roc:
                plot_roc(Y_test, y_pred)
