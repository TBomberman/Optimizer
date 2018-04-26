from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from helpers.utilities import all_stats
from helpers.callbacks import NEpochLogger

class Gene_Predictor():

    def __init__(self):
        model = Sequential()
        self.model = model

    def train(self, data, labels):
        neuron_count = data.shape[1]
        nb_classes = labels.shape[1]
        dropout = 0.2
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        out_epoch = NEpochLogger(display=5)

        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=0.7, test_size=0.3)
        X_val, X_test, Y_val, y_test = train_test_split(X_test, Y_test, train_size=0.5, test_size=0.5)

        self.model.add(Dense(neuron_count, input_shape=(neuron_count,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(neuron_count))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, Y_train, batch_size=2048, nb_epoch=10000,
                  verbose=0, validation_data=(X_test, Y_test), callbacks=[early_stopping, out_epoch])

        y_score_train = self.model.predict_proba(X_train)
        y_score_test = self.model.predict_proba(X_test)
        y_score_val = self.model.predict_proba(X_val)

        train_stats = all_stats(Y_train[:, 0], y_score_train[:, 0])
        test_stats = all_stats(Y_test[:, 0], y_score_test[:, 0], train_stats[-1])
        val_stats = all_stats(Y_val[:, 0], y_score_val[:, 0], train_stats[-1])

        print('Hidden layers: 2, Neurons per layer:', neuron_count)
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        print('Total:', ['{:6.2f}'.format(val) for val in [train_stats[0] + test_stats[0] + val_stats[0]]])

    def predict(self, data):
        return self.model.predict(data)