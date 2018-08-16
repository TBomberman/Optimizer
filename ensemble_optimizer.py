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

def do_optimize(nb_classes, data, labels, model_file_prefix=None, drug_ids=None, gene_ids=None):
    n = len(labels)
    labels = np_utils.to_categorical(labels, nb_classes)
    print("Train size:", n)

    model = MlpEnsemble(saved_models_path=model_file_prefix + '_ensemble_models/', patience=5, x_drug_ids=drug_ids,
                        x_gene_ids=gene_ids)
    if use_fit:
        model.fit(data, labels)

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