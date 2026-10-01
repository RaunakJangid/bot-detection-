"""XGBoost reference model (strong tabular baseline for the results table)."""

from __future__ import annotations

import time

import numpy as np
import torch
import xgboost as xgb

from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, confusion
from shield.utils.config import Paths
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)


def train_xgboost(paths: Paths, dataset: str, task: str, seed: int, cfg: dict, use_gpu: bool,
                  force: bool = False) -> dict:
    out = paths.outputs / "xgboost" / f"{dataset}_{task}_s{seed}"
    if (out / "metrics.json").exists() and not force:
        log.info("XGBoost %s exists, skipping", out.name)
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    data = load_processed(paths.processed(dataset), task)
    tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
    rng = np.random.default_rng(seed)
    cap = cfg.get("max_train_rows")
    idx = stratified_sample(tr.y, cap, rng, min_per_class=1000) if cap and len(tr.y) > cap else np.arange(len(tr.y))
    vidx = stratified_sample(va.y, min(len(va.y), 1_000_000), rng, min_per_class=500)

    # XGBoost needs contiguous labels: train on the classes present, give absent classes probability 0.
    present = np.unique(tr.y[idx])
    remap = np.full(data.n_classes, -1)
    remap[present] = np.arange(len(present))
    vidx = vidx[remap[va.y[vidx]] >= 0]
    binary = len(present) == 2
    clf = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
        tree_method="hist", device="cuda" if use_gpu else "cpu", random_state=seed,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss",
    )
    t0 = time.time()
    clf.fit(np.asarray(tr.X[idx]), remap[tr.y[idx]], eval_set=[(np.asarray(va.X[vidx]), remap[va.y[vidx]])],
            verbose=False)
    train_seconds = time.time() - t0

    # Predict through a DMatrix: avoids XGBoost's GPU-model/CPU-data mismatch fallback warning.
    booster = clf.get_booster()
    best = getattr(clf, "best_iteration", None)
    rng_iter = (0, best + 1) if best is not None else (0, 0)
    proba = np.zeros((len(te.y), data.n_classes))
    for i in range(0, len(te.y), 1_000_000):
        p = booster.predict(xgb.DMatrix(np.asarray(te.X[i:i + 1_000_000])), iteration_range=rng_iter)
        proba[i:i + 1_000_000, present] = np.stack([1 - p, p], 1) if binary else p
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    clf.save_model(out / "model.json")
    np.save(out / "confusion.npy", confusion(te.y, logits, data.n_classes))
    metrics = {"dataset": dataset, "task": task, "seed": seed, "train_rows": int(len(idx)),
               "classes_present": present,
               "best_iteration": int(getattr(clf, "best_iteration", cfg["n_estimators"])),
               "train_seconds": round(train_seconds, 1),
               "test": classification_metrics(te.y, logits, data.n_classes, data.classes, full=True),
               "hardware": hardware_record()}
    save_json(metrics, out / "metrics.json")
    log.info("XGBoost %s test macro-F1 %.4f", out.name, metrics["test"]["macro_f1"])
    return metrics


def gpu_available() -> bool:
    return torch.cuda.is_available()


def xgb_logits(run_dir, X: np.ndarray, n_classes: int, chunk: int = 1_000_000) -> np.ndarray:
    """Log-probabilities over all n_classes from a saved XGBoost run (absent classes get ~log 0)."""
    m = load_json(run_dir / "metrics.json")
    present = np.asarray(m["classes_present"])
    booster = xgb.Booster()
    booster.load_model(run_dir / "model.json")
    best = m.get("best_iteration")
    proba = np.zeros((len(X), n_classes))
    for i in range(0, len(X), chunk):
        p = booster.predict(xgb.DMatrix(np.asarray(X[i:i + chunk])),
                            iteration_range=(0, best + 1) if best is not None else (0, 0))
        proba[i:i + chunk][:, present] = np.stack([1 - p, p], 1) if len(present) == 2 else p
    return np.log(np.clip(proba, 1e-12, 1.0))
