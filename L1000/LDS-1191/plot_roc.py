import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from scipy import interp
from keras.utils import np_utils
from itertools import cycle

def plot_multi_class_roc(filenames):
    n_classes = len(filenames)
    data_folder_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/1vsall/"
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0, n_classes):
        y_true = np.load(data_folder_path + filenames[i] + "_true.npz")['arr_0']
        y_pred = np.load(data_folder_path + filenames[i] + "_pred.npz")['arr_0']
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, 1], y_pred[:, 1])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()

    class_names = {
        0: "No Regulation",
        1: "Downregulation",
        2: "Upregulation"
    }
    # plot them
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    # 'k--' is the black dashed line, lw = line width
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # set axis
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MCF7: ROC curve for ' + str(n_classes) + " classes")
    plt.legend(loc="lower right")
    plt.show()

filenames = {}
filenames[0] = "2018-09-18 05:01:02.533051"
filenames[1] = "2018-09-18 04:18:07.858064"
filenames[2] = "2018-09-18 03:03:10.649133"

def check_active_index():
    path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/load_data/10pBlindGapFixed/"
    float_filename = "MCF7_Multi_10b_10p_90g_test_npY_float.npz"
    class_filename = "MCF7_Multi_10b_10p_90g_test_npY_class.npz"
    floats = np.load(path+float_filename)['arr_0']
    classes = np.load(path + class_filename)['arr_0']
    test = 1

    labels = [1, 0]
    test = np_utils.to_categorical(labels, 2)
    labels = [0, 1]
    test = np_utils.to_categorical(labels, 2)
    test = 1
    # turns out that it's ordered by 0 is index 0 and 1 is index 1

# check_active_index()

def show_non_duplicates():
    from L1000.data_loader import get_feature_dict, load_csv
    path = "/data/datasets/gwoo/L1000/LDS-1191/Metadata/"
    filename = "Small_Molecule_Metadata.txt"
    csv = load_csv(path + filename)
    items = []
    for row in csv:
        items.append(row[8].strip())
    unique_list = list(set(items))
    for item in unique_list:
        print(item)

# plot_multi_class_roc(filenames)
# show_non_duplicates()