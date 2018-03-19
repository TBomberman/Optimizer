from random import sample

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.regularizers import l1, l1_l2, l2

import keras_enums as enums
from helpers.utilities import all_stats

# local variables
dropout = 0.2
dense = 500
batch_size = 2048
nb_epoch =1000 #1000 cutoff 1 #3000 cutoff  2 and
# regularizer = l1 # l1 beats the other 2
lammy = 0

def regularizer(lammy):
    return None

# for reproducibility
# np.random.seed(1337)
# random.seed(1337)

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
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    # Y_val = np_utils.to_categorical(y_val, nb_classes)

    for hyperparam in range(1, 10):
        # lammy = hyperparam * 0.00001
        lammy = 0.00003 # l1
        # neuron_count = dense * hyperparam
        neuron_count = dense
        layer_count = 1
        optimizer = enums.optimizers[6]  # rmsprop or adam
        activation = enums.activation_functions[9]
        activation_input = enums.activation_functions[8]
        activation_output = enums.activation_functions[3]

        model = Sequential()
        history = History()
        model.add(Dense(neuron_count, input_shape=(X_train.shape[1],), activity_regularizer=regularizer(lammy)))
        # model.add(Activation('tanh'))
        model.add(Activation(activation_input))
        model.add(Dropout(dropout))

        add_dense_dropout(layer_count, neuron_count, model, activation)

        if nb_classes > 2:
            model.add(Dense(labels.shape[1], activity_regularizer=regularizer(lammy)))
        else:
            model.add(Dense(1, activity_regularizer=regularizer(lammy)))
        # model.add(Activation('softmax'))
        model.add(Activation(activation_output))
        model.summary()

        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[history])

        score = model.evaluate(X_test, Y_test, verbose=0)

        y_score_train = model.predict_proba(X_train)
        y_score_test = model.predict_proba(X_test)
        y_score_val = model.predict_proba(X_val)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        if nb_classes > 2:
            Y_train = Y_train[:, 0]
            y_score_train = y_score_train[:, 0]
            Y_test = Y_test[:, 0]
            y_score_test = y_score_test[:, 0]
            Y_val = Y_val[:, 0]
            y_score_val = y_score_val[:, 0]

        train_stats = all_stats(Y_train, y_score_train)
        test_stats = all_stats(Y_test, y_score_test, train_stats[-1])
        val_stats = all_stats(Y_val, y_score_val, train_stats[-1])

        plt.scatter(Y_train, y_score_train)
        plt.draw()

        print_out = 'Hidden layers: %s, Neurons per layer: %s, Hyperparam: %s' % (layer_count, neuron_count, hyperparam)
        print(print_out)
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.2f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])
        # print(history.history.keys())
        # summarize history for loss

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

    plt.show()

def add_dense_dropout(count, neuron_count, model, activation):
    for x in range(0, count):
        model.add(Dense(neuron_count, activity_regularizer=regularizer(lammy)))
        model.add(Activation(activation))
        model.add(Dropout(dropout))