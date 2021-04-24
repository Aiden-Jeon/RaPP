import numpy as np
from sklearn import metrics


def get_auroc(label: np.ndarray, score: np.ndarray) -> float:
    fprs, tprs, _ = metrics.roc_curve(label, score)
    return metrics.auc(fprs, tprs)


def get_aupr(label: np.ndarray, score: np.ndarray) -> float:
    precisions, recalls, _ = metrics.precision_recall_curve(label, score)
    return metrics.auc(recalls, precisions)