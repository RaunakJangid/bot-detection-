"""Student -> expert cascade (v3 step 5).

Every flow is classified by the tiny student; flows whose student confidence (max softmax probability,
Chow's rule) is below a threshold are escalated to an expert (the teacher or the teacher ensemble),
whose decision replaces the student's. The threshold is set on VALIDATION, either for a target
escalation budget or as the smallest budget whose validation macro-F1 reaches the expert's.
"""

from __future__ import annotations

import numpy as np

from shield.eval.metrics import classification_metrics, softmax


def confidence(logits: np.ndarray) -> np.ndarray:
    return softmax(logits.astype(np.float64)).max(1)


def neg_entropy(logits: np.ndarray) -> np.ndarray:
    p = softmax(logits.astype(np.float64))
    return (p * np.log(np.clip(p, 1e-300, 1))).sum(1)


SCORES = {"confidence": confidence, "entropy": neg_entropy}   # higher = keep the student's decision


def cascade_predict(s_logits: np.ndarray, e_logits: np.ndarray, threshold: float,
                    score: str = "confidence") -> tuple[np.ndarray, np.ndarray]:
    """(predictions, escalated mask)."""
    esc = SCORES[score](s_logits) < threshold
    pred = s_logits.argmax(1)
    pred[esc] = e_logits[esc].argmax(1)
    return pred, esc


def threshold_for_budget(val_conf: np.ndarray, budget: float) -> float:
    """Threshold escalating (about) `budget` of the validation flows (0 -> never escalate)."""
    if budget <= 0:
        return -np.inf
    if budget >= 1:
        return np.inf
    return float(np.quantile(val_conf, budget))


def _f1(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    return classification_metrics(y, np.eye(n, dtype=np.float32)[pred], n)["macro_f1"]


def cascade_curve(s_val, e_val, y_val, s_test, e_test, y_test, n_classes: int, budgets, score: str = "confidence") -> list[dict]:
    """Per budget: val-chosen threshold, actual val/test escalation and macro-F1."""
    vc = SCORES[score](s_val)
    rows = []
    for b in budgets:
        t = threshold_for_budget(vc, b)
        pv, ev = cascade_predict(s_val, e_val, t, score)
        pt, et = cascade_predict(s_test, e_test, t, score)
        rows.append({"budget": float(b), "threshold": t, "val_escalated": float(ev.mean()),
                     "test_escalated": float(et.mean()), "val_macro_f1": _f1(y_val, pv, n_classes),
                     "test_macro_f1": _f1(y_test, pt, n_classes)})
    return rows


def matching_budget(curve: list[dict], expert_val_f1: float, tol: float = 0.005) -> dict:
    """The smallest budget whose VALIDATION macro-F1 is within `tol` of the expert's (else the best)."""
    ok = [r for r in curve if r["val_macro_f1"] >= expert_val_f1 - tol]
    return min(ok, key=lambda r: r["budget"]) if ok else max(curve, key=lambda r: r["val_macro_f1"])
