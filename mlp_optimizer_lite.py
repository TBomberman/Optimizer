import numpy as np
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
import keras_enums as enums
from helpers.utilities import all_stats
from helpers.callbacks import NEpochLogger
import sklearn.metrics as metrics
from keras.layers.normalization import BatchNormalization
import datetime

# local variables
dropout = 0.2
batch_size = 2**12
nb_epoch = 10000 #1000 cutoff 1 #3000 cutoff  2 and
train_percentage = 0.7
patience = 5

def do_optimize(nb_classes, data, labels):
    rtn_model = None
    d = data.shape[1]
    if nb_classes > 1:
        labels = np_utils.to_categorical(labels, nb_classes)

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True)
    sum_auc = 0
    count = 0
    sum_prec = 0
    sum_fscore = 0
    for train_index, test_index in kf.split(data):
        count += 1
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train = data[train_index]
        Y_train = labels[train_index]
        val_indexes, test_indexes = train_test_split(test_index, train_size=0.5, test_size=0.5, shuffle=True)
        X_val = data[val_indexes]
        Y_val = labels[val_indexes]
        X_test = data[test_indexes]
        Y_test = labels[test_indexes]

        for hyperparam in [0.00000001]:
            neuron_count = int(d)
            layer_count = 1
            optimizer = enums.optimizers[4]
            activation_input = enums.activation_functions[1]
            activation = enums.activation_functions[8]
            activation_output = enums.activation_functions[5]
            model = Sequential()
            model.add(Dense(neuron_count, input_shape=(neuron_count,), kernel_regularizer=l2(hyperparam)))
            model.add(BatchNormalization())
            model.add(Activation(activation_input))
            model.add(Dropout(dropout))
            add_dense_dropout(layer_count, neuron_count, model, activation, hyperparam)
            model.add(Dense(nb_classes, kernel_regularizer=l2(hyperparam)))
            model.add(BatchNormalization())
            model.add(Activation(activation_output))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            history = History()
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
            out_epoch = NEpochLogger(display=5)

            model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                      verbose=0, validation_data=(X_val, Y_val), callbacks=[history, early_stopping, out_epoch],
                      class_weight='auto')

            y_pred_train = model.predict_proba(X_train)
            y_pred_test = model.predict_proba(X_test)

            def print_stats(train_stats, test_stats):
                print_out = 'Hidden layers: %s, Neurons per layer: %s, Hyperparam: %s' % (
                layer_count + 1, neuron_count, hyperparam)
                print(print_out)
                print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff | Max F score')
                print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
                print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
                print('Total:', ['{:6.3f}'.format(val) for val in [train_stats[0] + test_stats[0]]])

            def save(ytrue, ypred):
                data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/1vsall/"
                prefix = str(datetime.datetime.now())
                print("saving", data_folder_path + prefix)
                np.savez(data_folder_path + prefix + "_pred", ypred)
                np.savez(data_folder_path + prefix + "_true", ytrue)

            def print_acc(text, Y_train, y_pred_train):
                y_pred = np.argmax(y_pred_train, axis=1)
                y_true = np.argmax(Y_train, axis=1)
                target_names = [0, 1, 2]
                cm = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                accs = cm.diagonal()
                print(text, "Accuracy class 0", accs[0])  # number of actual 0's predicted correctly
                print(text, "Accuracy class 1", accs[1])  # number of actual 1's predicted correctly
                print(text, "Accuracy class 2", accs[2])  # number of actual 2's predicted correctly

                report = metrics.classification_report(y_true, y_pred)
                print("Report", report)

            print_acc("Train", Y_train, y_pred_train)
            print_acc("Test", Y_test, y_pred_test)

            if nb_classes > 2:
                for class_index in range(0, nb_classes):
                    print('class', class_index, 'stats')
                    train_stats = all_stats(Y_train[:, class_index], y_pred_train[:, class_index])
                    test_stats = all_stats(Y_test[:, class_index], y_pred_test[:, class_index])
                    print_stats(train_stats, test_stats)
                    sum_auc += test_stats[0]
                    sum_prec += test_stats[4]
                    sum_fscore += test_stats[6]
            elif nb_classes == 2:
                train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
                test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1])
                save(Y_test[:, 1], y_pred_test[:, 1])
                print_stats(train_stats, test_stats)
                sum_auc += test_stats[0]
                sum_prec += test_stats[4]
                sum_fscore += test_stats[6]

            rtn_model = model
        print("running kfold auc", count, sum_auc / count, 'prec', sum_prec / count, 'fscore', sum_fscore / count)
    print("final kfold auc", sum_auc / n_splits, 'prec', sum_prec / count, 'fscore', sum_fscore / count)
    return rtn_model

def add_dense_dropout(count, neuron_count, model, activation, hyperparam):
    for x in range(0, count):
        model.add(Dense(neuron_count, kernel_regularizer=l2(hyperparam)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout))