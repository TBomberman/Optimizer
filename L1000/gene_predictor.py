from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from helpers.utilities import all_stats
from helpers.callbacks import NEpochLogger
from keras.utils import np_utils
from keras.models import model_from_json

def train_model(train_data, train_labels):
    model = Gene_Predictor()
    model.train(train_data, train_labels)
    return model.model

def load_model(file_prefix):
    # load json and create model
    json_file = open(file_prefix + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_prefix + '.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

def save_model(model, file_prefix):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_prefix + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_prefix + ".h5")
    print("Saved model", file_prefix)

class Gene_Predictor():

    def __init__(self):
        model = Sequential()
        self.model = model

    def train(self, data, labels):
        neuron_count = data.shape[1]
        nb_classes = len(set(labels))
        dropout = 0.2
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        out_epoch = NEpochLogger(display=5)

        cat_labels = np_utils.to_categorical(labels, nb_classes)
        X_train, X_test, Y_train, Y_test = train_test_split(data, cat_labels, train_size=0.7, test_size=0.3)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, train_size=0.5, test_size=0.5)

        self.model.add(Dense(neuron_count, input_shape=(neuron_count,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(neuron_count))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, Y_train, batch_size=2048, epochs=10000,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[early_stopping, out_epoch])

        y_score_train = self.model.predict_proba(X_train)
        y_score_test = self.model.predict_proba(X_test)
        y_score_val = self.model.predict_proba(X_val)

        train_stats = all_stats(Y_train[:, 0], y_score_train[:, 0])
        val_stats = all_stats(Y_val[:, 0], y_score_val[:, 0])
        test_stats = all_stats(Y_test[:, 0], y_score_test[:, 0], val_stats[-1])


        print('Hidden layers: 2, Neurons per layer:', neuron_count)
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.2f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])

    def predict(self, data):
        return self.model.predict(data)