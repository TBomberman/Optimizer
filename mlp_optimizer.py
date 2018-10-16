from random import sample

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils, multi_gpu_model
from keras.regularizers import l1, l1_l2, l2
from sklearn.model_selection import train_test_split, KFold
from helpers.plot_roc import plot_roc
import keras_enums as enums
from helpers.utilities import all_stats, scatter2D_plot
from helpers.callbacks import NEpochLogger
from keras.utils import plot_model
import datetime
import sklearn.metrics as metrics

# local variables
dropout = 0.2
dense = 1403
batch_size = 2**12
nb_epoch = 10000 #1000 cutoff 1 #3000 cutoff  2 and
regularizer = l1 # l1 beats the other 2
lammy = 0
use_plot = False
train_percentage = 0.85
patience = 5

# uncomment this to disable regularizer
def regularizer(lammy):
    return None

# for reproducibility
# np.random.seed(1337)
# random.seed(1337)

def save_model(model, file_prefix):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_prefix + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_prefix + ".h5")
    print("Saved model", file_prefix)

def do_optimize(nb_classes, data, labels, model_file_prefix=None, pos_class_weight=None):
    rtn_model = None
    n = len(labels)
    d = data.shape[1]
    if nb_classes > 1:
        labels = np_utils.to_categorical(labels, nb_classes)

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True)
    sum_auc = 0
    count = 0
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

        # train_size = int(train_percentage * n)
        # print("Train size:", train_size)
        # test_size = int((1-train_percentage) * n)
        # X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
        # # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
        # Y_train = y_train
        # Y_test = y_test
        # # Y_val = y_val

        # for hyperparam in range(4, 7):
        for hyperparam in [1]:
            lammy = 8 / (10**7) # LNCAP from LDS-1494 optimized
            # lammy = 0.0000001 # l1
            # neuron_count = dense * hyperparam
            neuron_count = int(d)# * 0.2 * hyperparam)

            # model = get_model(neuron_count, nb_classes, hyperparam)
            layer_count = 0
            optimizer = enums.optimizers[4]
            # act 0: 'elu', 1: 'selu', 2: 'sigmoid', 3: 'linear', 4: 'softplus', 5: 'softmax', 6: 'tanh', 7: 'hard_sigmoid',
            # 8: 'relu', 9: 'softsign'
            activation_input = enums.activation_functions[1]
            activation = enums.activation_functions[8]
            activation_output = enums.activation_functions[5]
            # usign rmse single output
            # activation_output = enums.activation_functions[2]

            model = Sequential()
            print('Patience', patience)

            model.add(Dense(neuron_count, input_shape=(neuron_count,), activity_regularizer=regularizer(lammy)))
            # model.add(Activation('tanh'))
            model.add(Activation(activation_input))
            model.add(Dropout(dropout))

            add_dense_dropout(layer_count, neuron_count, model, activation)

            model.add(Dense(nb_classes, activity_regularizer=regularizer(lammy)))
            # model.add(Activation('softmax'))
            model.add(Activation(activation_output))
            # model.summary() # too much verbage

            # multi_model = multi_gpu_model(model, gpus=6)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

            history = History()
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
            out_epoch = NEpochLogger(display=5)
            class_weight = { 1: pos_class_weight, 0: 1-pos_class_weight}

            model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                      verbose=0, validation_data=(X_val, Y_val), callbacks=[history, early_stopping, out_epoch],
                      class_weight='auto')
            # save_model(model, model_file_prefix)
            # score = model.evaluate(X_test, Y_test, verbose=0)
            #
            # print('Test score:', score[0])
            # print('Test accuracy:', score[1])
            # print("metrics", model.metrics_names)
            #
            # score = model.evaluate(X_cold, Y_cold)
            # print('Blind Test score:', score[0])
            # print('Blind Test accuracy:', score[1])

            y_pred_train = model.predict_proba(X_train)
            y_pred_test = model.predict_proba(X_test)
            # y_pred_val = model.predict_proba(X_val)
            # y_pred_cold = model.predict_proba(X_cold)

            def print_stats(train_stats, test_stats): #, val_stats):
                print_out = 'Hidden layers: %s, Neurons per layer: %s, Hyperparam: %s' % (
                layer_count + 1, neuron_count, hyperparam)
                print(print_out)
                print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
                print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
                print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])
                # print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
                print('Total:', ['{:6.3f}'.format(val) for val in [train_stats[0] + test_stats[0]]])# + val_stats[0]]])

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
                print(text, "Accuracy class 0", accs[0])
                print(text, "Accuracy class 1", accs[1])
                print(text, "Accuracy class 2", accs[2])

                report = metrics.classification_report(y_true, y_pred)
                print("Report", report)

            print_acc("Train", Y_train, y_pred_train)
            print_acc("Test", Y_test, y_pred_test)
            # print_acc("Test Blind", Y_cold, y_pred_cold)

            if nb_classes > 2:
                for class_index in range(0, nb_classes):
                    print('class', class_index, 'stats')
                    train_stats = all_stats(Y_train[:, class_index], y_pred_train[:, class_index])
                    # val_stats = all_stats(Y_val[:, class_index], y_pred_val[:, class_index])
                    test_stats = all_stats(Y_test[:, class_index], y_pred_test[:, class_index])
                    # blind_stats = all_stats(Y_cold[:, class_index], y_pred_cold[:, class_index])
                    print_stats(train_stats, test_stats) #, blind_stats)
                    sum_auc += test_stats[0]
            elif nb_classes == 2:
                train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
                # val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1] )
                save(Y_test[:, 1], y_pred_test[:, 1])
                test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1])#, val_stats[-1])
                print_stats(train_stats, test_stats)#, val_stats)
                sum_auc += test_stats[0]
            else:
                train_stats = all_stats(Y_train, y_pred_train)
                # val_stats = all_stats(Y_val, y_pred_val)
                test_stats = all_stats(Y_test, y_pred_test) #, val_stats[-1])
                print_stats(train_stats, test_stats)#, val_stats)

            # print(history.history.keys())
            # summarize history for loss

            if use_plot:
                plot_roc(Y_test[:,0], y_pred_test[:,0])
                # plt.scatter(Y_train, y_score_train)
                # plt.draw()

                # plot
                # nth = int(nb_epoch *0.05)
                nth = 1
                five_ploss = history.history['loss'][0::nth]
                five_pvloss = history.history['val_loss'][0::nth]
                plt.figure()
                plt.plot(five_ploss)
                plt.plot(five_pvloss)
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.draw()

            rtn_model = model
        print("running kfold auc", count, sum_auc / count)
    print("final kfold auc", sum_auc / n_splits)
    return rtn_model

def add_dense_dropout(count, neuron_count, model, activation):
    for x in range(0, count):
        model.add(Dense(neuron_count, activity_regularizer=regularizer(lammy)))
        model.add(Activation(activation))
        model.add(Dropout(dropout))

def get_model(neuron_count, nb_classes, hyperparam=0):
    layer_count = 1

    optimizer = enums.optimizers[4]
    # act 0: 'elu', 1: 'selu', 2: 'sigmoid', 3: 'linear', 4: 'softplus', 5: 'softmax', 6: 'tanh', 7: 'hard_sigmoid',
    # 8: 'relu', 9: 'softsign'
    activation_input = enums.activation_functions[1]
    activation = enums.activation_functions[8]
    activation_output = enums.activation_functions[5]
    # usign rmse single output
    # activation_output = enums.activation_functions[2]

    model = Sequential()
    print('Patience', patience)

    model.add(Dense(neuron_count, input_shape=(neuron_count,), activity_regularizer=regularizer(lammy)))
    # model.add(Activation('tanh'))
    model.add(Activation(activation_input))
    model.add(Dropout(dropout))

    add_dense_dropout(layer_count, neuron_count, model, activation)

    model.add(Dense(nb_classes, activity_regularizer=regularizer(lammy)))
    # model.add(Activation('softmax'))
    model.add(Activation(activation_output))
    # model.summary() # too much verbage

    # multi_model = multi_gpu_model(model, gpus=6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

def my_plot_model():
    model = get_model(1274, 2)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

my_plot_model()
