"""Strongest-boosting baselines under the strict split: LightGBM and XGBoost, each tuned on VALIDATION macro-F1
over a small grid (fit on a stratified train subsample for speed), then refit on up to --cap train rows with the
chosen setting; test scored once. Logits -> outputs_v3/logits (names lgbm_tuned_s0 / xgb_tuned_s0)."""

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("gbdt_strict")
LGB_GRID = {"num_leaves": [31, 127], "learning_rate": [0.05, 0.1], "min_data_in_leaf": [20, 200], "class_weight": [None, "sqrt"]}
XGB_GRID = {"max_depth": [8, 12], "learning_rate": [0.05, 0.1], "min_child_weight": [1, 10], "class_weight": [None, "sqrt"]}


def weights(y, mode):
    if mode is None:
        return None
    c = np.bincount(y).astype(float)
    w = 1 / np.sqrt(np.maximum(c, 1))
    return (w / (w * c).sum() * c.sum())[y]


def fit_lgb(p, X, y, Xv, yv, n):
    prm = {"objective": "multiclass" if n > 2 else "binary", "verbose": -1, "num_threads": 18, "seed": 0,
           "num_leaves": p["num_leaves"], "learning_rate": p["learning_rate"], "min_data_in_leaf": p["min_data_in_leaf"],
           "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1}
    if n > 2:
        prm["num_class"] = n
    m = lgb.train(prm, lgb.Dataset(X, y, weight=weights(y, p["class_weight"])), 3000,
                  valid_sets=[lgb.Dataset(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    def pred(Z):
        q = m.predict(Z, num_iteration=m.best_iteration)
        return np.log(np.clip(np.stack([1 - q, q], 1) if n == 2 else q, 1e-12, 1))
    return pred


def fit_xgb(p, X, y, Xv, yv, n):
    m = xgb.XGBClassifier(n_estimators=3000, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                          min_child_weight=p["min_child_weight"], tree_method="hist", device="cuda", random_state=0,
                          early_stopping_rounds=100, objective="multi:softprob" if n > 2 else "binary:logistic")
    m.fit(X, y, sample_weight=weights(y, p["class_weight"]), eval_set=[(Xv, yv)], verbose=False)
    return lambda Z: np.log(np.clip(m.predict_proba(Z), 1e-12, 1))


def main():
    p = parser(__doc__)
    p.add_argument("--tune-rows", type=int, default=1_000_000)
    p.add_argument("--cap", type=int, default=8_000_000)
    args = p.parse_args()
    paths = Paths(args.smoke)
    for task in ([args.task] if args.task else ["class34", "family", "binary"]):
        ds = args.dataset if args.dataset not in ("all", "main") else "ciciot_full"
        data = load_processed(paths.processed(ds), task)
        tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
        n, rng = data.n_classes, np.random.default_rng(0)
        ti = stratified_sample(tr.y, args.tune_rows, rng, 200)
        es = stratified_sample(va.y, 200_000, rng, 100)          # early-stopping slice of validation
        Xt, yt, Xe, ye, Xv = np.asarray(tr.X[ti]), tr.y[ti], np.asarray(va.X[es]), va.y[es], np.asarray(va.X)
        out = {}
        for name, grid, fit in (("lgbm_tuned", LGB_GRID, fit_lgb), ("xgb_tuned", XGB_GRID, fit_xgb)):
            res = []
            for vals in itertools.product(*grid.values()):
                prm = dict(zip(grid, vals))
                t0 = time.time()
                f1 = classification_metrics(va.y, fit(prm, Xt, yt, Xe, ye, n)(Xv), n)["macro_f1"]
                res.append((f1, prm))
                log.info("%s %s %s val %.4f (%.0fs)", ds, task, name, f1, time.time() - t0)
            best_f1, best = max(res, key=lambda r: r[0])
            fi = stratified_sample(tr.y, min(args.cap, len(tr.y)), rng, 1000) if len(tr.y) > args.cap else np.arange(len(tr.y))
            pred = fit(best, np.asarray(tr.X[fi]), tr.y[fi], Xe, ye, n)
            z = {"validation": pred(Xv), "test": pred(np.asarray(te.X))}
            lc = paths.v3 / "logits" / f"{ds}_{task}"
            lc.mkdir(parents=True, exist_ok=True)
            for s, v in z.items():
                np.save(lc / f"{name}_s0_{s}.npy", v.astype(np.float16))
            out[name] = {"params": best, "val_tune_subset": best_f1, "train_rows": int(len(fi)),
                         "val": classification_metrics(va.y, z["validation"], n)["macro_f1"],
                         "test": classification_metrics(te.y, z["test"], n, data.classes, full=True)}
            log.info("%s %s %s best %s -> val %.4f test %.4f", ds, task, name, best, out[name]["val"], out[name]["test"]["macro_f1"])
        save_json(out, paths.v3 / "gbdt_strict" / f"{ds}_{task}.json")


if __name__ == "__main__":
    main()
