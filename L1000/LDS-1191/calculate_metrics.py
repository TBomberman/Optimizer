import numpy as np
import sklearn.metrics as metrics
import os.path

def calculate():
    working_path = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/x10/warm/"
    folders = next(os.walk(working_path))[1]
    for folder in folders:
        # print(working_path + folder)
        try:
            y_pred_prob = None
            y_true_prob = None
            for i in range(0, 10):
                file_name = "EnsembleModel" + str(i) + "_y_pred.npz"
                file_name = working_path + folder + '/' + file_name
                if not os.path.isfile(file_name):
                    continue
                print('load', file_name)
                file = np.load(file_name)
                if y_pred_prob is None:
                    y_pred_prob = file.f.arr_0
                else:
                    y_pred_prob = np.concatenate((y_pred_prob, file.f.arr_0), axis=0)
                file_name = "EnsembleModel" + str(i) + "_y_val.npz"
                file_name = working_path + folder + '/' + file_name
                file = np.load(file_name)
                if y_true_prob is None:
                    y_true_prob = file.f.arr_0
                else:
                    y_true_prob = np.concatenate((y_true_prob, file.f.arr_0), axis=0)

            #classify them
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_true_prob, axis=1)

            accuracy = metrics.accuracy_score(y_true, y_pred)
            print("Accuracy", accuracy)

            target_names = [0, 1, 2]
            cm = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            accs = cm.diagonal()
            print("Accuracy class 0", accs[0])
            print("Accuracy class 1", accs[1])
            print("Accuracy class 2", accs[2])

            mcc = metrics.matthews_corrcoef(y_true, y_pred)
            print("MCC", mcc)

            report = metrics.classification_report(y_true, y_pred)
            print("Report", report)

        except:
            continue

def save_list():
    # list = ['m', 'y', 'n', 'a', 'm', 'e', 'i', 's', 'g', 'o', 'd', 'w', 'i', 'n']
    list = [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4,3,2,1]
    list = np.asarray(list)
    np.savez('test', list)
    loaded = np.load('test.npz')
    test = 1 + 1

calculate()
# save_list()

