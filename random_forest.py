import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from random import sample
from helpers.utilities import scatter2D_plot
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def do_optimize(nb_classes, data, labels, data_test=None, labels_test=None):
    if data_test is None:
        # ids
        work_ids = range(len(labels))
        train_ids = sample(work_ids, int(0.7 * len(work_ids)))
        test_ids = np.setdiff1d(work_ids, train_ids)

        # X data
        X_train = data[train_ids, :]
        X_test = data[test_ids, :]

        # Y data
        y_train = labels[train_ids]
        y_test = labels[test_ids]
    else:
        # ids
        test_ids = range(len(labels_test))

        # X data
        X_train = data
        X_test = data_test[test_ids, :]

        # Y data
        y_train = labels
        y_test = labels_test[test_ids]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = y_train
    Y_test = y_test

    sum_test = np.sum(Y_test)
    sum_train = np.sum(Y_train)
    show_roc = False

    if sum_train > 0 and sum_train < len(Y_train) and sum_test > 0 and sum_test < len(Y_test):

        model = RandomForestRegressor(n_estimators=100, min_samples_split=50, n_jobs=-1)
        # model = RandomForestClassifier()

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_train)
        # uncomment below for RandomForestClassifier
        # tr_error = np.mean((Y_train - y_pred)**2)
        # tr_error = np.mean(y_pred != Y_train)
        # scatter2D_plot(y_train, y_pred, "train", "rf")
        tr_auc = roc_auc_score(Y_train, y_pred)


        y_pred = model.predict(X_test)
        # uncomment below for RandomForestClassifier
        # te_error = np.mean((Y_test - y_pred) ** 2)
        # te_error = np.mean(y_pred != Y_test)
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
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            plt.title('Receiver Operating Characteristic')
            plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.2])
            plt.ylim([-0.1, 1.2])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
