"""Metrics shared by baselines, neural models and the final test evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and ROC-AUC for the binary task.

    Positive class = PNEUMONIA (1). `recall_pneumonia` is the PRIMARY metric;
    `recall_normal` is the specificity (how many healthy children are correctly cleared).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 and len(set(y_prob)) > 1 else float("nan")
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        recall_pneumonia=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_normal=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        precision_pneumonia=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        precision_normal=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        f1_pneumonia=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_normal=f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        auc=auc, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
