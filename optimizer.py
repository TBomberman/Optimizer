import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import History
from random import sample
from utilities import minmax, remove_constant_values, all_stats
import matplotlib.pyplot as plt

# local variables
dropout = 0.2
dense = 100
cutoff = 0.5
batch_size = 20000
nb_classes = 2
nb_epoch = 10000 #1000 cutoff 1 #3000 cutoff  2 and

np.random.seed(1337)  # for reproducibility
import random
random.seed(1337)

def do_optimize(dataset):
    # data set up
    data = dataset[:, 0:52]
    labels = dataset[:, -1]

    # Remove constant
    max_X = np.amax(data[:, :-1])
    min_X = np.amin(data[:, :-1])
    data[:, :-1] = (data[:, :-1] - min_X) / (max_X - min_X)
    data[np.where(data[:, 0:-1] == np.NAN), 0:-1] = 0

    y = np.zeros((len(labels), 1)).astype(int)
    pos_id = np.where(abs(labels) > cutoff)[0]
    y[pos_id] = 1
    work_id = range(len(y))

    train_partition = sample(work_id, int(0.7 * len(work_id)))
    test_partition = np.setdiff1d(work_id, train_partition)
    val_partition = test_partition[0:int(len(test_partition) * 0.5)]
    test_partition = np.setdiff1d(test_partition, val_partition)

    X_train = data[train_partition, :-1]
    X_test = data[test_partition, :-1]
    X_val = data[val_partition, :-1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')

    y_train = y[train_partition]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = y[test_partition]
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    y_val = y[val_partition]
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    for hyperparam in range(1, 10):
        # neuron_count = dense * hyperparam
        neuron_count = dense
        layer_count = hyperparam

        model = Sequential()
        history = History()
        model.add(Dense(neuron_count, input_shape=(X_train.shape[1],)))
        model.add(Activation('tanh'))
        model.add(Dropout(dropout))

        add_dense_dropout(layer_count, neuron_count, model)

        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[history])

        score = model.evaluate(X_test, Y_test, verbose=0)

        y_score_train = model.predict_proba(X_train)
        y_score_test = model.predict_proba(X_test)
        y_score_val = model.predict_proba(X_val)

        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])

        train_stats = all_stats(Y_train[:, 1], y_score_train[:, 1])
        test_stats = all_stats(Y_test[:, 1], y_score_test[:, 1], train_stats[-1])
        val_stats = all_stats(Y_val[:, 1], y_score_val[:, 1], train_stats[-1])

        print('Hidden layers and dropouts: %s, Neurons per layer: %s' % (layer_count, neuron_count))
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        # print(history.history.keys())
        # summarize history for loss


        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def add_dense_dropout(count, neuron_count, model):
    for x in range(0, count):
        model.add(Dense(neuron_count))
        model.add(Activation('tanh'))
        model.add(Dropout(dropout))