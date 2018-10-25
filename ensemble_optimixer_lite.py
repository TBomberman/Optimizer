from keras.utils import np_utils
from L1000.three_model_ensemble_lite import ThreeModelEnsemble

def do_optimize(nb_classes, data, labels, model_file_prefix=None):
    labels = np_utils.to_categorical(labels, nb_classes)
    model = ThreeModelEnsemble(saved_models_path=model_file_prefix + '_ensemble_models/', patience=5)
    model.fit(data, labels)
