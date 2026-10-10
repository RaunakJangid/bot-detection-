"""Tiny tree-ensemble students (what production / in-switch IDSs deploy) vs the neural edge students, at matched
storage. LightGBM, depth-limited, few boosting rounds, trained on the same strict train split (row cap), early
stopping on a VALIDATION subset, test scored once (raw argmax and with the validation-fitted per-class rule).
Storage model (deployable flat array): internal node = f32 threshold + u8 feature + 2 x u16 children = 9 B,
leaf = f32 value = 4 B; multiclass = one tree per class per round. Latency: 1 thread, batch 1 and 256."""

import os
import time

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("tiny_trees")
JOBS = [("ciciot_full", "class34"), ("ciciomt", "category"), ("ciciomt", "attack")]


def tree_bytes(booster):
    d = booster.dump_model()["tree_info"]
    leaves = sum(t["num_leaves"] for t in d)
    return 9 * (leaves - len(d)) + 4 * leaves, len(d)


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--rounds", type=int, nargs="*", default=[2, 5, 10, 20, 50])
    p.add_argument("--leaves", type=int, nargs="*", default=[4, 8, 16])
    p.add_argument("--cap", type=int, default=4_000_000)
    args = p.parse_args()
    import lightgbm as lgb
    paths = Paths(args.smoke)
    for ds, task in JOBS:
        out = paths.v3 / "tiny_trees" / f"{ds}_{task}.json"
        if out.exists() and not args.force:
            continue
        data = load_processed(paths.processed(ds), task)
        tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
        n = data.n_classes
        prior = np.bincount(tr.y, minlength=n) / len(tr.y)
        rng = np.random.default_rng(0)
        idx = np.arange(len(tr.y)) if len(tr.y) <= args.cap else stratified_sample(tr.y, args.cap, rng, 1000)
        vidx = stratified_sample(va.y, min(len(va.y), 300_000), rng, 200)
        cnt = np.bincount(tr.y[idx], minlength=n).astype(float)
        w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w / (w * cnt).sum() * cnt.sum()   # same weighting as students
        Xtr, Xv, Xt = np.asarray(tr.X[idx]), np.asarray(va.X), np.asarray(te.X)
        rows = []
        for L in args.leaves:
            params = {"objective": "multiclass", "num_class": n, "num_leaves": L, "learning_rate": 0.2,
                      "min_data_in_leaf": 50, "verbose": -1, "num_threads": 16, "seed": 0}
            dtr = lgb.Dataset(Xtr, tr.y[idx], weight=w[tr.y[idx]])
            dva = lgb.Dataset(Xv[vidx], va.y[vidx], reference=dtr)
            full = lgb.train(params, dtr, max(args.rounds), valid_sets=[dva])
            for R in args.rounds:
                zv = np.log(np.clip(full.predict(Xv, num_iteration=R), 1e-12, 1))
                zt = np.log(np.clip(full.predict(Xt, num_iteration=R), 1e-12, 1))
                bias = fit_decision_rule(zv, va.y, prior)["bias"]
                b = lgb.Booster(model_str=full.model_to_string(num_iteration=R))
                size, ntree = tree_bytes(b)
                os.environ["OMP_NUM_THREADS"] = "1"
                x1 = Xt[:2000]
                t0 = time.perf_counter()
                for i in range(len(x1)):
                    b.predict(x1[i:i + 1], num_threads=1)
                lat = (time.perf_counter() - t0) / len(x1) * 1e3
                t0 = time.perf_counter(); b.predict(Xt[:256 * 200], num_threads=1)
                thr = 256 * 200 / (time.perf_counter() - t0)
                r = {"leaves": L, "rounds": R, "trees": ntree, "bytes": size,
                     "val": classification_metrics(va.y, zv, n)["macro_f1"],
                     "test": classification_metrics(te.y, zt, n)["macro_f1"],
                     "val_rule": classification_metrics(va.y, apply_bias(zv, bias), n)["macro_f1"],
                     "test_rule": classification_metrics(te.y, apply_bias(zt, bias), n)["macro_f1"],
                     "latency_ms_b1": lat, "throughput_b256": thr}
                rows.append(r)
                log.info("%s_%s leaves %d rounds %d: %d B | test %.4f (+rule %.4f) | %.3f ms, %.0f flows/s", ds, task,
                         L, R, size, r["test"], r["test_rule"], lat, thr)
        save_json({"rows": rows}, out)


if __name__ == "__main__":
    main()
