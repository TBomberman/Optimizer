from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from itertools import cycle

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


def plot_precision_recall(y_true, y_pred, title='Precision Recall Curve'):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(title)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')

    p, r, _ = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    ax1.plot(r, p)
    ax2.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax2.plot([0, 1], [0, 1], 'r--')
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    plt.show()


def plot_roc_multi(y_true_list, y_pred_list, legend, title='Reciever Operating Characteristic'):
    n_classes = len(y_true_list)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0, n_classes):
        y_true = y_true_list[i]
        y_pred = y_pred_list[i]

        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()

    # plot them
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        label = legend[i] + ' (area = {0:0.2f})'.format(roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)

    # 'k--' is the black dashed line, lw = line width
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # set axis
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
