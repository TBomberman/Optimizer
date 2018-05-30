from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from helpers.plot_roc import plot_roc
from helpers.utilities import all_stats
from L1000.mlp_ensemble import MlpEnsemble
import numpy as np

train_percentage = 0.7
use_plot = False
use_fit = False
load_data = True

def do_optimize(nb_classes, data, labels):
    n = len(labels)
    labels = np_utils.to_categorical(labels, nb_classes)

    train_size = int(train_percentage * n)
    print("Train size:", train_size)
    test_size = int((1-train_percentage) * n)
    if load_data:
        X_train = np.load("X_train.npz")['arr_0'] # not balanced
        X_test = np.load("X_test.npz")['arr_0']
        y_train = np.load("y_train.npz")['arr_0']
        y_test = np.load("y_test.npz")['arr_0']
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
        np.savez("X_train", X_train)
        np.savez("X_test", X_test)
        np.savez("y_train", y_train)
        np.savez("y_test", y_test)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
    Y_train = y_train
    Y_test = y_test
    Y_val = y_val

    model = MlpEnsemble()
    if use_fit:
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    y_pred_val = model.predict_proba(X_val)

    train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
    val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1] )
    test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1], val_stats[-1])

    print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
    print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
    print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
    print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
    print('Total:', ['{:6.2f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])

    if use_plot:
        plot_roc(Y_test[:,1], y_pred_test[:,1])
