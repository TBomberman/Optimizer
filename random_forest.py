import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from random import sample
from helpers.utilities import scatter2D_plot

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

    # model = RandomForestRegressor()
    total_features = len(data[0])
    # for i in range(1,10):
    i = 1
    # num_features = int((total_features*i/10)**(1/2))
    min_samples_split = 2**i
    estimators = 10 * i
    max_depth = 5 * i
    model = RandomForestClassifier()

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)
    # tr_error = np.mean((Y_train - y_pred)**2)
    tr_error = np.mean(y_pred != Y_train)
    # scatter2D_plot(y_train, y_pred, "train", "rf")

    y_pred = model.predict(X_test)
    # te_error = np.mean((Y_test - y_pred) ** 2)
    te_error = np.mean(y_pred != Y_test)
    # scatter2D_plot(Y_test, y_pred, "test", "rf")

    y_pred = model.predict(X_val)
    # v_error = np.mean((Y_val - y_pred) ** 2)
    v_error = np.mean(y_pred != Y_val)
    # scatter2D_plot(Y_val, y_pred, "val", "rf")

    avg_test = (te_error + v_error)/2
    print("Random Forest (sklearn) error train: %.3f test: %.3f val: %.3f avg test: %.3f" % (tr_error, te_error, v_error, avg_test))

