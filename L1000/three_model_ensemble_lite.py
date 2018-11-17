import numpy as np
from mlp_optimizer_lite import do_optimize
import sklearn.metrics as metrics
from keras.utils import np_utils
import os

class ThreeModelEnsemble:

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

    def fit(self, x=None, y=None, batch_size=2**12, epochs=10000, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None,validation_steps=None, **kwargs):

        self.up_model_y = y[:,2]
        self.down_model_y = y[:,1]
        self.stable_model_y = y[:,0]
        up_model_y = self.up_model_y
        down_model_y = self.down_model_y
        stable_model_y = self.stable_model_y

        def train(direction, x, y):
            print('training', direction)
            model = do_optimize(2, x, y)
            file_prefix = self.saved_models_path + "1vsAll" + direction
            if self.save_models:
                self.save_model(model, file_prefix)
            return model

        self.up_model = train("Up", x, up_model_y)
        # self.down_model = train("Down", x, down_model_y)
        # self.stable_model = train("Stable", x, stable_model_y)


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
        mean = np.mean(self.up_model_y)
        std = np.std(self.up_model_y)
        y_pred_up = y_pred_up - mean / std

        y_pred_down = self.down_model.predict_proba(x)
        mean = np.mean(self.down_model_y)
        std = np.std(self.down_model_y)
        y_pred_down = y_pred_down - mean / std

        y_pred_stable = self.stable_model.predict_proba(x)
        mean = np.mean(self.stable_model_y)
        std = np.std(self.stable_model_y)
        y_pred_stable = y_pred_stable - mean / std

        y_pred = np.concatenate((np.reshape(y_pred_stable[:,1],(-1, 1)),
                                np.reshape(y_pred_down[:, 1], (-1, 1)),
                                np.reshape(y_pred_up[:, 1], (-1, 1))), axis=1)
        return y_pred

    def save_model(self, model, file_prefix):
        # serialize model to JSON
        model_json = model.to_json()
        os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
        with open(file_prefix + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(file_prefix + ".h5")
        print("Saved model", file_prefix, "to disk")