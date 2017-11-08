import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import History
from random import sample
from utilities import minmax, remove_constant_values, all_stats
import matplotlib.pyplot as plt
import keras_enums as enums
import random

# local variables
dropout = 0.2
dense = 100
batch_size = 10000
nb_epoch =20000 #1000 cutoff 1 #3000 cutoff  2 and

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
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    for hyperparam in range(1, 2):
        # neuron_count = dense * hyperparam
        neuron_count = dense
        layer_count = 5
        optimizer = enums.optimizers[1] #rmsprop
        activation = enums.activation_functions[0] #elu
        activation_input = enums.activation_functions[7] # hard signmoid
        activation_output = enums.activation_functions[2] # sigmoid

        model = Sequential()
        history = History()
        model.add(Dense(neuron_count, input_shape=(X_train.shape[1],)))
        # model.add(Activation('tanh'))
        model.add(Activation(activation_input))
        model.add(Dropout(dropout))

        add_dense_dropout(layer_count, neuron_count, model, activation)

        model.add(Dense(nb_classes))
        # model.add(Activation('softmax'))
        model.add(Activation(activation_output))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[history])

        score = model.evaluate(X_test, Y_test, verbose=0)

        y_score_train = model.predict_proba(X_train)
        y_score_test = model.predict_proba(X_test)
        y_score_val = model.predict_proba(X_val)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        train_stats = all_stats(Y_train[:, 1], y_score_train[:, 1])
        test_stats = all_stats(Y_test[:, 1], y_score_test[:, 1], train_stats[-1])
        val_stats = all_stats(Y_val[:, 1], y_score_val[:, 1], train_stats[-1])

        print('Hidden layers and dropouts: %s, Neurons per layer: %s' % (layer_count, neuron_count))
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
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
        model.add(Dense(neuron_count))
        model.add(Activation(activation))
        model.add(Dropout(dropout))