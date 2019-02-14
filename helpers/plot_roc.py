from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(y_true, y_pred, title='Receiver Operating Characteristic'):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

