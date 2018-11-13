import datetime
from ensemble_optimizer_lite import do_optimize
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import helpers.email_notifier as en
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.48
set_session(tf.Session(config=config))

# target_cell_names = ['PC3', 'HT29']
# target_cell_names = ['MCF7', 'A375']
# target_cell_names = ['VCAP', 'A549']
target_cell_names = ['MCF7']
direction = 'Multi' #'Down'
load_data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/morgan2048/5p/"
data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/cv/morgan2048/5p/"
gap_factors = [0.0]
percentiles = [5]
class_weights = [0.01]
for target_cell_name in target_cell_names:
    for bin in [10]:
        for percentile_down in percentiles:
            for gap_factor in gap_factors:
                for class_0_weight in class_weights:
                    file_suffix = target_cell_name + '_' + direction + '_' + str(bin) + 'b_' + \
                                  str(percentile_down) + 'p_' + str(int(gap_factor*100)) + 'g'
                    model_file_prefix = data_folder_path + str(datetime.datetime.now()) + '_' + file_suffix + \
                                        '_' + str(int(class_0_weight*100)) + 'c'
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
                        do_optimize(len(np.unique(npY_class)), npX, npY_class, model_file_prefix)
                    finally:
                        en.notify()