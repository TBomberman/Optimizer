import numpy as np
import sklearn.metrics as metrics

prefix = "/data/datasets/gwoo/L1000/LDS-1191/ensemble_models/PC3_Multi10b_p10_ensemble_models/"
file_name = "EnsembleModel0_y_pred.npz"
file = np.load(prefix + file_name)
y_pred_prob = file.f.arr_0
file_name = "EnsembleModel0_y_val.npz"
file = np.load(prefix + file_name)
y_true_prob = file.f.arr_0

#classify them
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_true_prob, axis=1)

accuracy = metrics.accuracy_score(y_true, y_pred)
print("Accuracy", accuracy)

mcc = metrics.matthews_corrcoef(y_true, y_pred)
print("MCC", mcc)

report = metrics.classification_report(y_true, y_pred)
print("Report", report)
