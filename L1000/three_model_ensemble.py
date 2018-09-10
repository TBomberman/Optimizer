from keras.engine.training import Model
from keras.models import Sequential, model_from_json
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.metrics import binary_accuracy
from helpers.callbacks import NEpochLogger
from pathlib import Path
import numpy as np
import os
# from L1000.mlp_ensemble import MlpEnsemble
from sklearn.model_selection import train_test_split
from mlp_optimizer import do_optimize
import sklearn.metrics as metrics
from keras.utils import np_utils
from sklearn import preprocessing

class ThreeModelEnsemble():

    def __init__(self, layers=None, name=None, patience=5, log_steps=5, dropout=0.2,
                 input_activation='selu', hidden_activation='relu', output_activation='softmax', optimizer='adam',
                 saved_models_path='ensemble_models/3model/', save_models=True, train_percentage=0.7):
        self.patience = patience
        self.dropout = dropout
        self.log_steps = log_steps
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.train_percentage = train_percentage

        # if models are saved, load them
        self.up_model = None
        self.down_model = None
        self.stable_model = None
        self.saved_models_path = saved_models_path
        self.save_models = save_models
        if save_models:
            return

        self.up_model = self.load_model(saved_models_path + "1vsAllUp")
        self.down_model = self.load_model(saved_models_path + "1vsAllDown")
        self.stable_model = self.load_model(saved_models_path + "1vsAllStable")

    def load_model(self, file_prefix):
        file = Path(file_prefix + '.json')
        if not file.exists():
            return None

        # load json and create model
        json_file = open(file_prefix + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(file_prefix + '.h5')
        print("Loaded model", file_prefix, "from disk")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return loaded_model

    def save_model(self, model, file_prefix):
        # serialize model to JSON
        model_json = model.to_json()
        os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
        with open(file_prefix + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(file_prefix + ".h5")
        print("Saved model", file_prefix, "to disk")

    def fit(self, x=None, y=None, batch_size=2**12, epochs=10000, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None,validation_steps=None, **kwargs):

        up_model_y = y[:,2]
        down_model_y = y[:,1]
        stable_model_y = y[:,0]

        def train(direction, x, y, pos_class_weight):
            print('training', direction)
            model = do_optimize(2, x, y, pos_class_weight=pos_class_weight)
            file_prefix = self.saved_models_path + "1vsAll" + direction
            if self.save_models:
                self.save_model(model, file_prefix)
            return model

        self.up_model = train("Up", x, up_model_y, class_weight[2])
        self.down_model = train("Down", x, down_model_y, class_weight[1])
        self.stable_model = train("Stable", x, stable_model_y, class_weight[0])

    def evaluate(self, x=None, y=None, batch_size=None, verbose=0, sample_weight=None, steps=None):
        up_model_y = y[:, 2]
        down_model_y = y[:, 1]
        stable_model_y = y[:, 0]

        sum_scores = 0
        def get_score(model, x, y):
            labels = np_utils.to_categorical(y, 2)
            score = model.evaluate(x, labels, verbose=0)
            if isinstance(score, list):
                return score[0]
            else:
                return score
        sum_scores += get_score(self.up_model, x, up_model_y)
        sum_scores += get_score(self.down_model, x, down_model_y)
        sum_scores += get_score(self.stable_model, x, stable_model_y)
        avg_score = sum_scores / 3

        y_pred = self.predict_proba(x)
        # standardize, with and without
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y, y_pred)
        return [avg_score, acc]

    def predict_proba(self, x):
        y_pred_up = self.up_model.predict_proba(x)
        y_pred_up = preprocessing.scale(y_pred_up)

        y_pred_down = self.down_model.predict_proba(x)
        y_pred_down = preprocessing.scale(y_pred_down)

        y_pred_stable = self.stable_model.predict_proba(x)
        y_pred_stable = preprocessing.scale(y_pred_stable)

        y_pred = np.concatenate((np.reshape(y_pred_stable[:,1],(-1, 1)),
                                np.reshape(y_pred_down[:, 1], (-1, 1)),
                                np.reshape(y_pred_up[:, 1], (-1, 1))), axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

