from random import sample

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils, multi_gpu_model
from keras.regularizers import l1, l1_l2, l2
from sklearn.model_selection import train_test_split
from helpers.plot_roc import plot_roc
import keras_enums as enums
from helpers.utilities import all_stats
from helpers.callbacks import NEpochLogger

# local variables
dropout = 0.2
dense = 1403
batch_size = 2**12
nb_epoch = 10000 #1000 cutoff 1 #3000 cutoff  2 and
regularizer = l1 # l1 beats the other 2
lammy = 0
use_plot = False
train_percentage = 0.7
patience = 10

# uncomment this to disable regularizer
def regularizer(lammy):
    return None

# for reproducibility
# np.random.seed(1337)
# random.seed(1337)

def do_optimize(nb_classes, data, labels, iid_validate_set_keys=None):
    n = len(labels)
    d = data.shape[1]
    if nb_classes > 1:
        labels = np_utils.to_categorical(labels, nb_classes)
    if iid_validate_set_keys is None:
        train_size = int(train_percentage * n)
        print("Train size:", train_size)
        test_size = int((1-train_percentage) * n)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
        Y_train = y_train
        Y_test = y_test
        Y_val = y_val
    else:
        set_keys = np.array(iid_validate_set_keys)
        key_set = set(iid_validate_set_keys)
        num_keys = len(key_set)
        test_keys_size = int(num_keys*(1-train_percentage))
        test_ids = []
        i = 0
        for key in key_set:
            if i > test_keys_size:
                break
            test_ids += np.where(set_keys == key)[0].tolist()
            i += 1
        mask = np.zeros(n, dtype=bool)
        mask[test_ids] = True
        test_ids, val_ids = train_test_split(test_ids, train_size=0.5, test_size=0.5)
        X_train = data[~mask]
        Y_train = labels[~mask]
        X_test = data[test_ids]
        Y_test = labels[test_ids]
        X_val = data[val_ids]
        Y_val = labels[val_ids]

    # for hyperparam in range(4, 7):
    for hyperparam in [1]:
        lammy = 8 / (10**7) # LNCAP from LDS-1494 optimized
        # lammy = 0.0000001 # l1
        # neuron_count = dense * hyperparam
        neuron_count = int(d)# * 0.2 * hyperparam)

        # model = get_model(neuron_count, nb_classes, hyperparam)
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

        history = History()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        out_epoch = NEpochLogger(display=5)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[history, early_stopping, out_epoch])

        score = model.evaluate(X_test, Y_test, verbose=0)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print("metrics", model.metrics_names)
        my_evaluate(X_test, Y_test, model)

        y_pred_train = model.predict_proba(X_train)
        y_pred_test = model.predict_proba(X_test)
        y_pred_val = model.predict_proba(X_val)

        if nb_classes > 1:
            train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
            val_stats = all_stats(Y_val[:, 1], y_pred_val[:, 1] )
            test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1], val_stats[-1])
        else:
            train_stats = all_stats(Y_train, y_pred_train)
            val_stats = all_stats(Y_val, y_pred_val)
            test_stats = all_stats(Y_test, y_pred_test, val_stats[-1])

        print_out = 'Hidden layers: %s, Neurons per layer: %s, Hyperparam: %s' % (layer_count + 1, neuron_count, hyperparam)
        print(print_out)
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff')
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.2f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])
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


def add_dense_dropout(count, neuron_count, model, activation):
    for x in range(0, count):
        model.add(Dense(neuron_count, activity_regularizer=regularizer(lammy)))
        model.add(Activation(activation))
        model.add(Dropout(dropout))

def my_evaluate(x, y, model):
    y_pred = model.predict_proba(x)
    y_pred[np.where(y_pred >= 0.5)] = 1
    y_pred[np.where(y_pred < 0.5)] = 0
    acc = np.mean(y_pred == y)
    print('My Test accuracy:', acc)

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