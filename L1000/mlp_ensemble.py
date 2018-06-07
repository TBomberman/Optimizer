from keras.engine.training import Model
from keras.models import Sequential, model_from_json
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.metrics import binary_accuracy
from helpers.callbacks import NEpochLogger
from pathlib import Path
import numpy as np
import os

class MlpEnsemble(Model):
    def __init__(self, layers=None, name=None, n_estimators=20, patience=20, log_steps=5, dropout=0.2,
                 input_activation='selu', hidden_activation='relu', output_activation='softmax', optimizer='adam',
                 saved_models_path='ensemble_models/'):
        self.patience = patience
        self.dropout = dropout
        self.log_steps = log_steps
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.n_estimators = n_estimators
        # if models are saved, load them
        self.models = {}
        self.saved_models_path = saved_models_path
        for i in range(0, n_estimators):
            file_prefix = saved_models_path + "EnsembleModel" + str(i)
            file = Path(file_prefix + '.json')
            if file.exists():
                self.models[file_prefix] = self.load_model(file_prefix)

    def load_model(self, file_prefix):
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

    def build_model(self, d):
        n_neuron = int(d)
        model = Sequential()
        model.add(Dense(n_neuron, input_shape=(n_neuron,)))
        model.add(Activation(self.input_activation))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_neuron))
        model.add(Activation(self.hidden_activation))
        model.add(Dropout(self.dropout))
        model.add(Dense(2))
        model.add(Activation(self.output_activation))
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        return model

    def fit(self, x=None, y=None, batch_size=2**12, epochs=10000, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None,validation_steps=None, save_models=True, **kwargs):

        # split the data into 20 sets
        # create array of n_estimator models
        # fit each model with each set
        # add option to save each model
        history = History()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=1, mode='auto')
        print('Patience', self.patience)
        out_epoch = NEpochLogger(display=self.log_steps)
        self.d = x.shape[1]

        all_train_indices = self.get_balanced_data_indices_sets(y)
        all_val_indices = self.get_balanced_data_indices_sets(validation_data[1])
        for i in range(0, self.n_estimators):
            train_indices = all_train_indices[i]
            val_indices = all_val_indices[i]
            file_prefix = self.saved_models_path + "EnsembleModel" + str(i)
            model = self.build_model(self.d)
            model.fit(x[train_indices], y[train_indices], batch_size=batch_size, epochs=epochs, verbose=0,
                      validation_data=(validation_data[0][val_indices], validation_data[1][val_indices]),
                      callbacks=[history, early_stopping, out_epoch])
            self.models[file_prefix] = model
            if save_models:
                self.save_model(model, file_prefix)

    def get_balanced_data_indices_sets(self, y):
        indices_set = []
        positive_idx = np.where(y[:,1])
        negative_idx = np.where(y[:,0])
        n_positives = int(np.sum(y[:,1]))
        n_negatives = int(np.sum(y[:,0]))
        print("positives", n_positives)
        print("negatives", n_negatives)
        for i in range(0, self.n_estimators):
            negative_locations_sample = np.random.randint(n_negatives, size=n_positives)
            negative_locations_sample = [negative_idx[0][negative_locations_sample]]
            combined_locations = [np.concatenate((positive_idx[0], negative_locations_sample[0]))]
            indices_set.append(combined_locations)
        return indices_set

    def evaluate(self, x=None, y=None, batch_size=None, verbose=0, sample_weight=None, steps=None):
        sum_scores = 0
        y_probs = []
        for name, model in self.models.items():
            print("collecting probabilities from", name)
            score = model.evaluate(x, y, verbose=0)
            if isinstance(score, list):
                sum_scores += score[0]
            else:
                sum_scores += score
            y_prob = model.predict_proba(x)
            y_prob[np.where(y_prob >= 0.5)] = 1 # use consensus
            y_prob[np.where(y_prob < 0.5)] = 0 # use consensus
            y_probs.append(y_prob)
        avg_score = sum_scores / self.n_estimators
        y_probs = np.asarray(y_probs)
        avg_y_probs = np.mean(y_probs, axis=0)
        avg_y_probs[np.where(avg_y_probs >= 0.5)] = 1
        avg_y_probs[np.where(avg_y_probs < 0.5)] = 0
        y_pred = avg_y_probs
        y = y.astype('float32')
        acc = np.mean(y == y_pred)
        return [avg_score, acc]

    def predict_proba(self, x):
        y_probs = []
        for name, model in self.models.items():
            y_prob = model.predict_proba(x)
            y_probs.append(y_prob)
        y_probs = np.asarray(y_probs)
        return np.mean(y_probs, axis=0)
