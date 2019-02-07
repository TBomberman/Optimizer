from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
import numpy as np

load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1484/load_data/morgan2048/"
npX = np.load(load_data_folder_path + "LNCAP_Up_10b_5p_npX.npz")['arr_0']
npY_class = np.load(load_data_folder_path + "LNCAP_Up_10b_5p_npY_class.npz")['arr_0']

x_train, x_test, y_train, y_test = train_test_split(npX, npY_class, train_size=0.7, test_size=0.3)

X = x_train
y = y_train
X_val = x_test
y_val = y_test

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,1024),
                    'dropout3': hp.uniform('dropout3', .25,.75)}
                    ]),

            'units1': int(3267*hp.uniform('units1', 0,1)),
            'units2': int(3267*hp.uniform('units2', 0,1)),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim=X.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best: ')
print(best)
