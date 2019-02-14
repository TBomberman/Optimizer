import datetime
from mlp_optimizer import do_optimize
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import helpers.email_notifier as en
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

# target_cell_names = ['PC3', 'HT29']
# target_cell_names = ['MCF7', 'A375']
# target_cell_names = ['VCAP', 'A549']
target_cell_names = ['VCAP']
direction = 'Multi' #'Down'
load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1484/load_data/morgan2048/"
data_folder_path = "/data/datasets/gwoo/L1000/LDS-1484/saved_models/"
gap_factors = [0.0]
percentiles = [5]
class_weights = [0.01]
for target_cell_name in target_cell_names:
    for bin in [10]:
        for percentile_down in percentiles:
            for gap_factor in gap_factors:
                for class_0_weight in class_weights:
                    # file_suffix = target_cell_name + '_' + direction + '_' + str(bin) + 'b_' + \
                    #               str(percentile_down) + 'p_' + str(int(gap_factor*100)) + 'g_all35Blind'
                    file_suffix = 'LNCAP_Down_10b_5p_3h_repeat'
                    model_file_prefix = data_folder_path + str(datetime.datetime.now()) + '_' + file_suffix # + \
                                        # '_' + str(int(class_0_weight*100)) + 'c'
                    print('load location', load_data_folder_path)
                    print('save location', model_file_prefix)
                    npX = np.load(load_data_folder_path + file_suffix + "_npX.npz")['arr_0']
                    npY_class = np.load(load_data_folder_path + file_suffix + "_npY_class.npz")['arr_0']
                    # cold_ids = np.load(load_data_folder_path + file_suffix + "_cold_ids.npz")['arr_0']
                    cold_ids = []

                    # for testing
                    # ints = random.sample(range(1,100000), 1000)
                    # npX = npX[ints]
                    # npY_class = npY_class[ints]
                    # cold_ids = cold_ids[ints]

                    try:
                        def save_model(model, file_prefix):
                            # serialize model to JSON
                            model_json = model.to_json()
                            os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
                            with open(file_prefix + ".json", "w") as json_file:
                                json_file.write(model_json)
                            model.save_weights(file_prefix + ".h5")
                            print("Saved model", file_prefix, "to disk")

                        do_optimize(len(np.unique(npY_class)), npX, npY_class, model_file_prefix)
                        # save_model(model, model_file_prefix)

                    finally:
                        en.notify()