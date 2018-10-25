from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from helpers.plot_roc import plot_roc
from helpers.utilities import all_stats, scatter2D_plot
from L1000.three_model_ensemble import ThreeModelEnsemble
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

train_percentage = 0.8
use_plot = False
use_fit = True
load_data = False
save_data = False

def do_optimize(nb_classes, data, labels, model_file_prefix=None, class_0_weight=None, cold_ids=None, labels_float=None,
                test_data=None):
    n = len(labels)
    labels = np_utils.to_categorical(labels, nb_classes)
    # test_data[1] = np_utils.to_categorical(test_data[1], nb_classes)

    # unique_cold_ids = np.unique(cold_ids)
    # n_unique_cold_ids = len(unique_cold_ids)
    # n_cold_test_ids = int((n_unique_cold_ids * (1 - train_percentage))/2)
    # cold_ids_for_test = np.random.choice(unique_cold_ids, n_cold_test_ids, replace=False)
    #
    # warm_indexes = []
    # cold_indexes = []
    #
    # for i in range(0, n):
    #     if cold_ids[i] in cold_ids_for_test:
    #         cold_indexes.append(i)
    #     else:
    #         warm_indexes.append(i)

    # if test_data is None:
    #     test_size = 1/9
    #     train_size = 1 - test_size
    #     warm_train_indexes, warm_test_indexes = train_test_split(warm_indexes, train_size=train_size, test_size=test_size, shuffle=True)
    #     X_train = data[warm_train_indexes]
    #     Y_train = labels[warm_train_indexes]
    #     X_test = data[warm_test_indexes]
    #     Y_test = labels[warm_test_indexes]
    # else:
    #     X_train = data
    #     Y_train = labels
    #     X_test = test_data[0]
    #     Y_test = test_data[1]

    X_train = data
    Y_train = labels

    # load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/blind/"
    # split = model_file_prefix.split("/")
    # split = split[len(split)-1].split("_")
    # target_cell_name = split[1]
    # X_cold = np.load(load_data_folder_path + target_cell_name + "_npX.npz")['arr_0']
    # Y_cold = np.load(load_data_folder_path + target_cell_name + "_npY_class.npz")['arr_0']
    # Y_cold = np_utils.to_categorical(Y_cold, nb_classes)

    # X_cold = data[cold_indexes]
    # Y_cold = labels[cold_indexes]

    # wrong = 0
    # for ind in warm_indexes:
    #     if (cold_ids[ind] in cold_ids_for_test):
    #         wrong += 1

    model = ThreeModelEnsemble(saved_models_path=model_file_prefix + '_ensemble_models/', patience=5)
    if class_0_weight is None:
        class_weight = None
    else:
        class_1_weight = (1-class_0_weight)/2
        class_weight = {
            0: class_0_weight,
            1: class_1_weight,
            2: class_1_weight
        }
    if use_fit:
        model.fit(X_train, Y_train, class_weight=class_weight)

    # score = model.evaluate(X_test, Y_test)
    # print('Warm Test score:', score[0])
    # print('Warm Test accuracy:', score[1])
    #
    # score = model.evaluate(X_test, Y_test)
    # print('Warm Val score:', score[0])
    # print('Warm Val accuracy:', score[1])
    #
    # score = model.evaluate(X_cold, Y_cold)
    # print('Blind Test score:', score[0])
    # print('Blind Test accuracy:', score[1])
    #
    # y_pred_train = model.predict_proba(X_train)
    # y_pred_test = model.predict_proba(X_test)
    # # y_pred_val = model.predict_proba(X_test)
    # # y_pred_val = y_pred_test
    # y_pred_cold = model.predict_proba(X_cold)
    #
    # # np.savez(model_file_prefix + "_warm_val_indexes", warm_val_indexes)
    # # np.savez(model_file_prefix + "_y_pred_warm_val", y_pred_val)
    # np.savez(model_file_prefix + "_y_pred_blind_test", y_pred_cold)

    def print_stats(train_stats, test_stats, val_stats, test_stats_cold):
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
        print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
        print('All stats test blind:', ['{:6.3f}'.format(val) for val in test_stats_cold])

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

    # if nb_classes > 2:
    #     print_acc("Train", Y_train, y_pred_train)
    #     print_acc("Test", Y_test, y_pred_test)
    #     # print_acc("Val", Y_test, y_pred_val)
    #     print_acc("Test Cold", Y_cold, y_pred_cold)
    #
    #     for class_index in range(0, nb_classes):
    #         print('class', class_index, 'stats')
    #         train_stats = all_stats(Y_train[:, class_index], y_pred_train[:, class_index])
    #         # val_stats = all_stats(Y_test[:, class_index], y_pred_val[:, class_index])
    #         test_stats = all_stats(Y_test[:, class_index], y_pred_test[:, class_index])
    #         test_stats_cold = all_stats(Y_cold[:, class_index], y_pred_cold[:, class_index])
    #         print_stats(train_stats, test_stats, test_stats, test_stats_cold)
    # elif nb_classes == 2:
    #     train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
    #     val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1])
    #     test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1], val_stats[-1])
    #     print_stats(train_stats, test_stats, val_stats)
    # else:
    #     train_stats = all_stats(Y_train, y_pred_train)
    #     val_stats = all_stats(Y_val, y_pred_val)
    #     test_stats = all_stats(Y_test, y_pred_test, val_stats[-1])
    #     print_stats(train_stats, test_stats, val_stats)

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

    model = ThreeModelEnsemble(saved_models_path=saved_models_path, save_models=False)

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