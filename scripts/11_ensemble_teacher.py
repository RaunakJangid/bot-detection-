"""v3 step 1: a stronger KD teacher = validation-weighted ensemble of the 5 ResMLP teachers and XGBoost.

    p = w * mean_seeds softmax(ResMLP) + (1 - w) * p_XGBoost,   w in {0, 0.1, ..., 1} chosen on validation
    macro-F1 (w = 1 / w = 0 recover the deep ensemble / XGBoost alone, so the choice is never worse on val).
Writes outputs_v3/teacher_ens/<dataset>_<task>/metrics.json and a logits-only KD cache (log p on train).
"""

import numpy as np
import torch

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed, predict_logits
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.eval.predictions import model_logits, run_dir
from shield.kd.cache import cache_ok, write_logit_cache
from shield.kd.trainer import ensemble_cache_dir
from shield.models.train import load_teacher
from shield.models.xgb import xgb_logits
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("ensemble")
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)


def macro_f1(y, p, n):
    return classification_metrics(y, np.log(np.clip(p, 1e-12, 1)), n)["macro_f1"]


def mixture(paths, data, seeds, split, device, tm):
    """(mean ResMLP softmax over seeds, XGBoost probabilities) on a cached split."""
    deep = np.mean([softmax(model_logits(paths, data, "teacher", s, (split,), device, tm)[split]) for s in seeds], 0)
    xg = np.exp(model_logits(paths, data, "xgboost", 0, (split,), device, tm)[split])
    return deep, xg


def train_mixture(paths, data, seeds, w, device, tm, chunk=2_000_000):
    """The same mixture on the training split (not cached: only the ensemble's log p is kept)."""
    X = data.splits["train"].X
    p = np.zeros((len(X), data.n_classes), np.float32)
    if w > 0:
        for s in seeds:
            net, _ = load_teacher(run_dir(paths, data.dataset, data.task, "teacher", s, tm), device)
            for i in range(0, len(X), chunk):
                p[i:i + chunk] += w / len(seeds) * softmax(predict_logits(net, X[i:i + chunk], device, 65536, amp=True))
            del net
            torch.cuda.empty_cache()
    if w < 1:
        xrun = run_dir(paths, data.dataset, data.task, "xgboost", 0, tm)
        for i in range(0, len(X), chunk):
            p[i:i + chunk] += (1 - w) * np.exp(xgb_logits(xrun, X[i:i + chunk], data.n_classes)).astype(np.float32)
    return p


def main():
    args = parser(__doc__).parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    tcfg = load_config("teacher", args.smoke)
    tm = tcfg["model"]
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main,full")]:
        seeds = reg.seeds(ds, tcfg["seeds"])
        for task in reg.tasks(ds, args.task):
            cache = ensemble_cache_dir(paths, ds, task)
            if cache_ok(cache) and not args.force:
                log.info("ensemble %s_%s exists, skipping", ds, task)
                continue
            data = load_processed(paths.processed(ds), task)
            n, yv, yt = data.n_classes, data.splits["validation"].y, data.splits["test"].y
            dv, xv = mixture(paths, data, seeds, "validation", device, tm)
            val = {float(w): macro_f1(yv, w * dv + (1 - w) * xv, n) for w in WEIGHTS}
            w = max(val, key=val.get)   # ties -> the smaller w (more XGBoost); dict keeps grid order
            dt, xt = mixture(paths, data, seeds, "test", device, tm)
            pt = w * dt + (1 - w) * xt
            prior = np.bincount(data.splits["train"].y, minlength=n) / len(data.splits["train"].y)
            rule = fit_decision_rule(np.log(np.clip(w * dv + (1 - w) * xv, 1e-12, 1)), yv, prior, device)
            lt = np.log(np.clip(pt, 1e-12, 1))
            metrics = {
                "dataset": ds, "task": task, "weight_resmlp": w, "val_macro_f1_by_weight": val,
                "val_macro_f1": {"teacher_s0": macro_f1(yv, softmax(model_logits(paths, data, "teacher", seeds[0],
                                                                                 ("validation",), device, tm)["validation"]), n),
                                 "deep_ensemble": val[1.0], "xgboost": val[0.0], "ensemble": val[w]},
                "test": classification_metrics(yt, lt, n, data.classes, full=True),
                "test_bias": classification_metrics(yt, apply_bias(lt, rule["bias"]), n, data.classes, full=True),
                "test_members": {"deep_ensemble": classification_metrics(yt, np.log(np.clip(dt, 1e-12, 1)), n),
                                 "xgboost": classification_metrics(yt, np.log(np.clip(xt, 1e-12, 1)), n)},
                "rule": {k: rule[k] for k in ("tau", "bias")},
            }
            save_json(metrics, cache.parent / "metrics.json")
            lc = paths.v3 / "logits" / f"{ds}_{task}"   # same cache layout as eval.predictions
            np.save(lc / "ensemble_s0_validation.npy", np.log(np.clip(w * dv + (1 - w) * xv, 1e-12, 1)).astype(np.float16))
            np.save(lc / "ensemble_s0_test.npy", lt.astype(np.float16))
            log.info("%s_%s: w_resmlp %.1f | val F1 deep %.4f xgb %.4f ens %.4f | test F1 ens %.4f (+bias %.4f)",
                     ds, task, w, val[1.0], val[0.0], val[w], metrics["test"]["macro_f1"],
                     metrics["test_bias"]["macro_f1"])
            ptr = train_mixture(paths, data, seeds, w, device, tm)
            write_logit_cache(np.log(np.clip(ptr, 1e-6, 1)), pt.argmax(1), cache, weight_resmlp=w)
            del ptr


if __name__ == "__main__":
    main()
