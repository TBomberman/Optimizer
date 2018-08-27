from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from helpers.plot_roc import plot_roc
from helpers.utilities import all_stats, scatter2D_plot
from L1000.mlp_ensemble import MlpEnsemble
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

train_percentage = 0.7
use_plot = False
use_fit = True
load_data = False
save_data = False

def do_optimize(nb_classes, data, labels, model_file_prefix=None):
    n = len(labels)
    labels = np_utils.to_categorical(labels, nb_classes)

    train_size = int(train_percentage * n)
    print("Train size:", train_size)
    test_size = int((1-train_percentage) * n)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
    Y_train = y_train
    Y_test = y_test
    Y_val = y_val

    model = MlpEnsemble(saved_models_path=model_file_prefix + '_ensemble_models/', patience=5)
    if use_fit:
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), class_weight='auto')

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    y_pred_val = model.predict_proba(X_val)

    y_prob = model.predict_proba(X_val)
    np.savez(model_file_prefix + "_x_val", X_val)
    np.savez(model_file_prefix + "_y_val", Y_val)
    np.savez(model_file_prefix + "_y_pred", y_prob)

    def print_stats(train_stats, test_stats, val_stats):
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
        print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.3f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])

    def print_acc(text, Y_train, y_pred_train):
        y_pred = np.argmax(y_pred_train, axis=1)
        y_true = np.argmax(Y_train, axis=1)
        target_names = [0, 1, 2]
        cm = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accs = cm.diagonal()
        print(text, "Accuracy class 0", accs[0])
        print(text, "Accuracy class 1", accs[1])
        print(text, "Accuracy class 2", accs[2])

        report = metrics.classification_report(y_true, y_pred)
        print("Report", report)

    if nb_classes > 2:
        print_acc("Train", Y_train, y_pred_train)
        print_acc("Test", Y_test, y_pred_test)
        print_acc("Val", Y_val, y_pred_val)

        for class_index in range(0, nb_classes):
            print('class', class_index, 'stats')
            train_stats = all_stats(Y_train[:, class_index], y_pred_train[:, class_index])
            val_stats = all_stats(Y_val[:, class_index], y_pred_val[:, class_index])
            test_stats = all_stats(Y_test[:, class_index], y_pred_test[:, class_index])
            print_stats(train_stats, test_stats, val_stats)
    elif nb_classes == 2:
        train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
        val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1])
        test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1], val_stats[-1])
        print_stats(train_stats, test_stats, val_stats)
    else:
        train_stats = all_stats(Y_train, y_pred_train)
        val_stats = all_stats(Y_val, y_pred_val)
        test_stats = all_stats(Y_test, y_pred_test, val_stats[-1])
        print_stats(train_stats, test_stats, val_stats)

    if use_plot:
        if nb_classes > 2:
            for class_index in range(0, nb_classes):
                plot_roc(Y_test[:, class_index], y_pred_test[:, class_index])
        elif nb_classes == 2:
            plot_roc(Y_test[:, 1], y_pred_test[:, 1])
        else:
            plot_roc(Y_test, y_pred_test)
        # scatter2D_plot(Y_train, y_pred_train)
        # scatter2D_plot(y_test, y_pred_test)

def evaluate(nb_classes, data, labels, file_prefix):
    saved_models_path = file_prefix + '_ensemble_models/'

    labels = np_utils.to_categorical(labels, nb_classes)
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

    def print_stats(test, pred):
        test_stats = all_stats(test, pred)
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
        print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
        if use_plot:
            plot_roc(test, pred)

    if nb_classes > 2:
        for class_index in range(0, nb_classes):
            print_stats(y_test[:, class_index], y_pred[:, class_index])

    elif nb_classes == 2:
        print_stats(y_test[:, 1], y_pred[:, 1])
    else:
        print_stats(y_test, y_pred)