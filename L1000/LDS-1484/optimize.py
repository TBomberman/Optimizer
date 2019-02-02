from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from hyperopt.hp import choice
from hyperopt.hp import uniform

import hyperas.optim as optim
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l1_l2, l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, EarlyStopping
from helpers.callbacks import NEpochLogger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1484/load_data/morgan2048/"
    npX = np.load(load_data_folder_path + "LNCAP_Up_10b_5p_npX.npz")['arr_0']
    npY_class = np.load(load_data_folder_path + "LNCAP_Up_10b_5p_npY_class.npz")['arr_0']

    x_train, x_test, y_train, y_test = train_test_split(npX, npY_class, train_size=0.85, test_size=0.15)

    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    def add_dense_dropout(count, neuron_count, model_to_add, activation, dropout, act_reg, kern_reg):
        for x in range(0, count):
            model_to_add.add(Dense(neuron_count, activity_regularizer=act_reg, kernel_regularizer=kern_reg))
            model_to_add.add(BatchNormalization())
            model_to_add.add(Activation(activation))
            model_to_add.add(Dropout(dropout))

    input_size = 3267
    input_activation = {{choice(['elu', 'selu', 'sigmoid', 'linear', 'softplus', 'softmax', 'tanh', 'hard_sigmoid',
                                 'relu', 'softsign'])}}
    inner_activation = {{choice(['elu', 'selu', 'sigmoid', 'linear', 'softplus', 'softmax', 'tanh', 'hard_sigmoid',
                                 'relu', 'softsign'])}}
    global_dropout = {{uniform(0, 1)}}
    n_hidden_layers = int(5 * {{uniform(0, 1)}})
    act_regularizer = {{choice([l1, l1_l2, l2])}}
    act_lambda_power = 10*{{uniform(0, 1)}}
    act_reg_lambda = 1/(10**act_lambda_power)
    kern_regularizer = {{choice([l1, l1_l2, l2])}}
    kern_lambda_power = 10 * {{uniform(0, 1)}}
    kern_reg_lambda = 1 / (10 ** kern_lambda_power)

    nb_epoch = 10000
    patience = 5

    model = Sequential()
    model.add(Dense(input_size, input_shape=(input_size,), activity_regularizer=act_regularizer(act_reg_lambda),
                    kernel_regularizer=kern_regularizer(kern_reg_lambda)))
    model.add(BatchNormalization())
    model.add(Activation(input_activation))
    model.add(Dropout(global_dropout))

    add_dense_dropout(n_hidden_layers, input_size, model, inner_activation, global_dropout,
                      act_regularizer(act_reg_lambda), kern_regularizer(kern_reg_lambda))

    model.add(Dense(2, input_shape=(input_size,), activity_regularizer=act_regularizer(act_reg_lambda),
                    kernel_regularizer=kern_regularizer(kern_reg_lambda)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])}})
    history = History()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    out_epoch = NEpochLogger(display=5)
    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128, 256, 1024, 2048])}},
              epochs=nb_epoch,
              verbose=0,
              validation_split=3/17,
              callbacks=[history, early_stopping, out_epoch],
              class_weight='auto')

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def run_bayesian_optimization():
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=280,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


run_bayesian_optimization()