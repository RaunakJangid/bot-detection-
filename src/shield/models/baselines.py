"""Small classical models of roughly the student's size: a depth-limited decision tree and logistic
regression. They answer the reviewer question "why not just a tiny tree / linear model?"."""

from __future__ import annotations

import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, confusion
from shield.utils.config import Paths
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)


def build(kind: str, cfg: dict, seed: int):
    if kind == "dtree":
        return DecisionTreeClassifier(max_depth=cfg["dtree_max_depth"], random_state=seed)
    if kind == "logreg":
        return LogisticRegression(max_iter=cfg["logreg_max_iter"], random_state=seed)
    raise ValueError(f"Unknown baseline {kind!r}")


def n_params(clf, kind: str) -> int:
    if kind == "dtree":
        t = clf.tree_
        internal = int((t.children_left >= 0).sum())
        return 2 * internal + (t.node_count - internal)  # (feature, threshold) per split + class per leaf
    return int(clf.coef_.size + clf.intercept_.size)


def full_proba(clf, X: np.ndarray, n_classes: int, chunk: int = 1_000_000) -> np.ndarray:
    out = np.zeros((len(X), n_classes))
    for i in range(0, len(X), chunk):
        out[i:i + chunk][:, clf.classes_] = clf.predict_proba(np.asarray(X[i:i + chunk]))
    return out


def cpu_latency(clf, n_features: int, runs: int = 200) -> dict:
    out = {}
    with threadpool_limits(1):
        for b in (1, 256):
            x = np.random.default_rng(0).normal(size=(b, n_features)).astype(np.float32)
            for _ in range(10):
                clf.predict(x)
            t = np.empty(runs)
            for i in range(runs):
                t0 = time.perf_counter(); clf.predict(x); t[i] = time.perf_counter() - t0
            out[f"latency_b{b}_ms"] = float(np.median(t)) * 1e3
            out[f"throughput_b{b}"] = b / float(np.median(t))
    return out


def train_baseline(paths: Paths, dataset: str, task: str, kind: str, cfg: dict, seed: int = 0,
                   force: bool = False) -> dict:
    out = paths.outputs / "baselines" / f"{dataset}_{task}_{kind}"
    if (out / "metrics.json").exists() and not force:
        log.info("Baseline %s exists, skipping", out.name)
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    data = load_processed(paths.processed(dataset), task, splits=("train", "test"))
    tr, te = data.splits["train"], data.splits["test"]
    cap = cfg["logreg_max_train_rows"] if kind == "logreg" else cfg["max_train_rows"]
    rng = np.random.default_rng(seed)
    idx = stratified_sample(tr.y, cap, rng, min_per_class=100) if len(tr.y) > cap else np.arange(len(tr.y))
    clf = build(kind, cfg, seed)
    t0 = time.time()
    clf.fit(np.asarray(tr.X[idx]), tr.y[idx])
    train_seconds = time.time() - t0
    logits = np.log(np.clip(full_proba(clf, te.X, data.n_classes), 1e-12, 1.0))
    np.save(out / "confusion.npy", confusion(te.y, logits, data.n_classes))
    blob = pickle.dumps(clf)
    with open(out / "model.pkl", "wb") as f:
        f.write(blob)
    metrics = {"dataset": dataset, "task": task, "model": kind, "train_rows": int(len(idx)),
               "train_seconds": round(train_seconds, 1), "params": n_params(clf, kind),
               "size_kb": round(len(blob) / 1024, 2), **cpu_latency(clf, data.n_features),
               "test": classification_metrics(te.y, logits, data.n_classes, data.classes, full=True),
               "hardware": hardware_record()}
    save_json(metrics, out / "metrics.json")
    log.info("Baseline %s test macro-F1 %.4f, %d params", out.name, metrics["test"]["macro_f1"], metrics["params"])
    return metrics
