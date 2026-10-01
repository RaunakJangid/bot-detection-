from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def classification_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int,
                           classes: list[str] | None = None, full: bool = False) -> dict:
    y_pred = logits.argmax(1)
    labels = np.arange(n_classes)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    if not full:
        return out
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = classes or [str(i) for i in labels]
    out["per_class"] = {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, n in enumerate(names)}
    present = np.unique(y_true)
    try:
        if n_classes == 2 and len(present) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, softmax(logits)[:, 1]))
        elif len(present) == n_classes:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, softmax(logits), multi_class="ovr",
                                                           average="macro", labels=labels))
    except ValueError:
        pass
    return out


def confusion(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, logits.argmax(1), labels=np.arange(n_classes))
