from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from helpers.plot_roc import plot_roc
from helpers.utilities import all_stats, scatter2D_plot
from L1000.mlp_ensemble import MlpEnsemble
import numpy as np
import matplotlib.pyplot as plt

train_percentage = 0.7
use_plot = False
use_fit = True
load_data = False
save_data = False

def do_optimize(nb_classes, data, labels, model_file_prefix=None):
    path_prefix = 'L1000/LDS-1191/saved_xy_data/'
    if load_data:
        data = np.load(path_prefix + "PC3npXEndsAllCutoffs.npz")['arr_0'] # not balanced
        labels = np.load(path_prefix + "PC3npY_classEndsAllCutoffs.npz")['arr_0']

    n = len(labels)
    # labels = np_utils.to_categorical(labels, nb_classes)

    train_size = int(train_percentage * n)
    print("Train size:", train_size)
    test_size = int((1-train_percentage) * n)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)

    if save_data:
        np.savez("X_train", X_train)
        np.savez("X_test", X_test)
        np.savez("y_train", y_train)
        np.savez("y_test", y_test)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
    Y_train = y_train
    Y_test = y_test
    Y_val = y_val

    model = MlpEnsemble(saved_models_path=model_file_prefix + '_ensemble_models/', patience=5)
    if use_fit:
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_pred = model.predict_proba(X_test)
    y_pred[np.where(y_pred >= 0.5)] = 1
    y_pred[np.where(y_pred < 0.5)] = 0
    acc = np.mean(y_pred == Y_test)
    print('My Test accuracy:', acc)

    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    y_pred_val = model.predict_proba(X_val)

    y_pred_train = np.reshape(y_pred_train, (-1))
    train_stats = all_stats(Y_train, y_pred_train)
    y_pred_val = np.reshape(y_pred_val, (-1))
    val_stats = all_stats(Y_val, y_pred_val)
    y_pred_test = np.reshape(y_pred_test, (-1))
    test_stats = all_stats(Y_test, y_pred_test)  # , val_stats[-1])

    # train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
    # val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1] )
    # test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1], val_stats[-1])

    print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
    print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
    print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
    print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
    print('Total:', ['{:6.3f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])

    if use_plot:
        # plot_roc(Y_test[:,0], y_pred_test[:,0])
        scatter2D_plot(Y_train, y_pred_train)
        scatter2D_plot(y_test, y_pred_test)

def evaluate(nb_classes, data, labels, file_prefix):
    saved_models_path = file_prefix + '_ensemble_models/'

    # labels = np_utils.to_categorical(labels, nb_classes)
    x_test = data
    y_test = labels

    model = MlpEnsemble(saved_models_path=saved_models_path, save_models=False)

    score = model.evaluate(x_test, y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_pred = model.predict_proba(x_test)
    # y_pred[np.where(y_pred >= 0.5)] = 1
    # y_pred[np.where(y_pred < 0.5)] = 0
    # acc = np.mean(y_pred == y_test)
    # print('My Test accuracy:', acc)

    # test_stats = all_stats(y_test[:, 1], y_pred[:, 1])
    y_pred = np.reshape(y_pred, (-1))
    test_stats = all_stats(y_test, y_pred)

    print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
    print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])

    if use_plot:
        scatter2D_plot(y_test, y_pred)
        # plot_roc(y_test[:, 1], y_pred[:, 1])
