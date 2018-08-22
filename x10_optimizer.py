from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from L1000.mlp_x10 import MlpX10

train_percentage = 0.9999999999999999

def do_optimize(nb_classes, data, labels, model_file_prefix=None, cold_ids=None):
    n = len(labels)
    labels = np_utils.to_categorical(labels, nb_classes)

    train_size = int(train_percentage * n)
    test_size = int((1-train_percentage) * n)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
    print('train samples', len(labels))

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)
    Y_test = y_test

    model = MlpX10(saved_models_path=model_file_prefix + '_x10_models/', patience=5, x_cold_ids=cold_ids)
    model.fit(X_train, y_train, validation_data=(X_test, Y_test))
