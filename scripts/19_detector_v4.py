"""v3 phase 2: stronger detectors, every choice on VALIDATION, test scored once.

Per main task:
  lgbm, catboost        gradient boosting with early stopping on validation (catboost on GPU)
  qt_ce_s*, qt_bs_s*    ResMLP (teacher.yaml size) on quantile-transformed inputs (heavy-tailed flow
                        features -> normal scores; transformer fitted on train only), trained with
                        cross-entropy or Balanced Softmax (Ren et al., NeurIPS 2020: logits + log prior)
  ens4                  ensemble selection (Caruana et al., ICML 2004): greedy forward selection with
                        replacement of model probability vectors, maximising validation macro-F1
Logits go to the shared v3 cache (outputs_v3/logits), so 10_posthoc.py / 14_cascade.py can use them.
Writes outputs_v3/detector_v4/<dataset>_<task>.json and detector_v4/summary.csv.
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import Batcher, load_processed, predict_logits, stratified_sample
from shield.engine import fit
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.eval.predictions import model_logits
from shield.models.teacher import build_teacher
from shield.models.train import train_cfg_for
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything

log = get_logger("detector_v4")
SPLITS = ("validation", "test")


def cache_path(paths, ds, task, name, split):
    return paths.v3 / "logits" / f"{ds}_{task}" / f"{name}_{split}.npy"


def have(paths, ds, task, name):
    return all(cache_path(paths, ds, task, name, s).exists() for s in SPLITS)


def save(paths, ds, task, name, logits: dict):
    for s, z in logits.items():
        f = cache_path(paths, ds, task, name, s)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, z.astype(np.float16))


def load(paths, ds, task, name):
    return {s: np.load(cache_path(paths, ds, task, name, s)).astype(np.float32) for s in SPLITS}


def train_rows(y, cap, seed):
    return np.arange(len(y)) if len(y) <= cap else stratified_sample(y, cap, np.random.default_rng(seed), 1000)


def run_lgbm(data, cap, seed):
    import lightgbm as lgb
    tr, va = data.splits["train"], data.splits["validation"]
    idx = train_rows(tr.y, cap, seed)
    vidx = train_rows(va.y, 500_000, seed)
    n = data.n_classes
    params = {"objective": "binary" if n == 2 else "multiclass", "num_class": 1 if n == 2 else n,
              "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": seed, "num_threads": 16}
    if n == 2:
        params.pop("num_class")
    dtr = lgb.Dataset(np.asarray(tr.X[idx]), tr.y[idx])
    dva = lgb.Dataset(np.asarray(va.X[vidx]), va.y[vidx], reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    out = {}
    for s in SPLITS:
        p = m.predict(np.asarray(data.splits[s].X), num_iteration=m.best_iteration)
        p = np.stack([1 - p, p], 1) if n == 2 else p
        out[s] = np.log(np.clip(p, 1e-12, 1))
    return out


def run_catboost(data, cap, seed):
    from catboost import CatBoostClassifier
    tr, va = data.splits["train"], data.splits["validation"]
    idx, vidx = train_rows(tr.y, cap, seed), train_rows(va.y, 500_000, seed)
    m = CatBoostClassifier(loss_function="Logloss" if data.n_classes == 2 else "MultiClass", iterations=4000,
                           learning_rate=0.1, depth=8, task_type="GPU", random_seed=seed, od_type="Iter",
                           od_wait=150, verbose=False)
    m.fit(np.asarray(tr.X[idx]), tr.y[idx], eval_set=(np.asarray(va.X[vidx]), va.y[vidx]), use_best_model=True)
    return {s: np.log(np.clip(m.predict_proba(np.asarray(data.splits[s].X)), 1e-12, 1)) for s in SPLITS}


def run_qt_resmlp(data, Xq: dict, prior, loss_kind, seed, tcfg_all, ds, device):
    seed_everything(seed)
    tcfg = train_cfg_for(tcfg_all, ds)
    tr = data.splits["train"]
    model = build_teacher("resmlp", data.n_features, data.n_classes, tcfg_all)
    log_prior = torch.as_tensor(np.log(np.clip(prior, 1e-12, None)), dtype=torch.float32, device=device)
    if loss_kind == "bs":   # Balanced Softmax: train on logits + log prior, predict with the raw logits
        loss_fn = lambda m, b: nn.functional.cross_entropy(m(b[0]) + log_prior, b[1])
    else:
        w = 1.0 / np.sqrt(np.maximum(np.bincount(tr.y, minlength=data.n_classes), 1))   # teacher.yaml: inverse_sqrt
        w = torch.as_tensor(w / (w * np.bincount(tr.y, minlength=data.n_classes)).sum() * len(tr.y),
                            dtype=torch.float32, device=device)
        loss_fn = lambda m, b: nn.functional.cross_entropy(m(b[0]), b[1], weight=w)
    batcher = Batcher(Xq["train"], tr.y, tcfg["batch_size"], device, balance_power=tcfg["balance_power"],
                      samples_per_epoch=tcfg["samples_per_epoch"], seed=seed)
    fit(model, batcher, Xq["validation"], data.splits["validation"].y, loss_fn, tcfg, device, data.n_classes,
        f"{ds}_{data.task}_qt_{loss_kind}_s{seed}")
    del batcher
    return {s: predict_logits(model, Xq[s], device, 65536, amp=bool(tcfg.get("amp"))) for s in SPLITS}


def macro_f1(y, p, n):
    return classification_metrics(y, np.log(np.clip(p, 1e-12, 1)), n)["macro_f1"]


def ensemble_selection(probs_val: dict, y, n, rounds=25):
    """Caruana et al. 2004: greedy forward selection with replacement; returns {member: count}."""
    names = list(probs_val)
    counts, cur, best = {}, None, -1.0
    for _ in range(rounds):
        cand = []
        for m in names:
            mix = probs_val[m] if cur is None else (cur * sum(counts.values()) + probs_val[m]) / (sum(counts.values()) + 1)
            cand.append((macro_f1(y, mix, n), m, mix))
        f, m, mix = max(cand, key=lambda t: t[0])
        if f <= best + 1e-6 and cur is not None:
            break
        best, cur = f, mix
        counts[m] = counts.get(m, 0) + 1
    return counts, best


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--gbdt-cap", type=int, default=6_000_000)
    args = p.parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    tcfg_all = load_config("teacher", args.smoke)
    rows = []
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main,full")]:
        for task in reg.tasks(ds, args.task):
            out = paths.v3 / "detector_v4" / f"{ds}_{task}.json"
            if out.exists() and not args.force:
                rows.extend(load_json(out)["rows"])
                continue
            data = load_processed(paths.processed(ds), task)
            n, yv, yt = data.n_classes, data.splits["validation"].y, data.splits["test"].y
            prior = np.bincount(data.splits["train"].y, minlength=n) / len(data.splits["train"].y)
            for name, fn in (("lgbm_s0", lambda: run_lgbm(data, args.gbdt_cap, 0)),
                             ("catboost_s0", lambda: run_catboost(data, args.gbdt_cap, 0))):
                if not have(paths, ds, task, name):
                    t0 = time.time()
                    save(paths, ds, task, name, fn())
                    log.info("%s_%s %s done in %.0fs", ds, task, name, time.time() - t0)
            need_qt = [f"qt_{k}_s{s}" for k in ("ce", "bs") for s in args.seeds if not have(paths, ds, task, f"qt_{k}_s{s}")]
            if need_qt:
                tr_idx = train_rows(data.splits["train"].y, 1_000_000, 0)
                qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                                         random_state=0).fit(np.asarray(data.splits["train"].X[tr_idx]))
                Xq = {s: qt.transform(np.asarray(data.splits[s].X)).astype(np.float32) for s in ("train", *SPLITS)}
                for name in need_qt:
                    kind, seed = name.split("_")[1], int(name.split("_s")[-1])
                    save(paths, ds, task, name, run_qt_resmlp(data, Xq, prior, kind, seed, tcfg_all, ds, device))
                    torch.cuda.empty_cache()
                del Xq
            members = {"teacher": [model_logits(paths, data, "teacher", s, SPLITS, device) for s in range(5)],
                       "xgboost": [model_logits(paths, data, "xgboost", 0, SPLITS, device)],
                       "lgbm": [load(paths, ds, task, "lgbm_s0")], "catboost": [load(paths, ds, task, "catboost_s0")],
                       "qt_ce": [load(paths, ds, task, f"qt_ce_s{s}") for s in args.seeds],
                       "qt_bs": [load(paths, ds, task, f"qt_bs_s{s}") for s in args.seeds]}
            # members average their own seeds (seed ensembles); val and test probabilities
            P = {m: {s: np.mean([softmax(z[s]) for z in zs], 0) for s in SPLITS} for m, zs in members.items()}
            single = {m: {"val": macro_f1(yv, P[m]["validation"], n), "test": macro_f1(yt, P[m]["test"], n)} for m in P}
            counts, val_ens = ensemble_selection({m: P[m]["validation"] for m in P}, yv, n)
            tot = sum(counts.values())
            pe = {s: sum(c * P[m][s] for m, c in counts.items()) / tot for s in SPLITS}
            le = {s: np.log(np.clip(pe[s], 1e-12, 1)) for s in SPLITS}
            save(paths, ds, task, "ens4_s0", le)
            rule = fit_decision_rule(le["validation"], yv, prior, device)
            test_ens = classification_metrics(yt, le["test"], n, data.classes, full=True)
            test_rule = classification_metrics(yt, apply_bias(le["test"], rule["bias"]), n, data.classes, full=True)
            recs = [{"dataset": ds, "task": task, "model": m, "val_macro_f1": v["val"], "test_macro_f1": v["test"]}
                    for m, v in single.items()]
            recs += [{"dataset": ds, "task": task, "model": "ens4", "val_macro_f1": val_ens,
                      "test_macro_f1": test_ens["macro_f1"], "members": counts},
                     {"dataset": ds, "task": task, "model": "ens4+rule", "val_macro_f1": rule["val_macro_f1_bias"],
                      "test_macro_f1": test_rule["macro_f1"], "members": counts}]
            save_json({"rows": recs, "test_ens4": test_ens, "test_ens4_rule": test_rule, "rule_bias": rule["bias"]}, out)
            log.info("%s_%s: %s", ds, task, {r["model"]: round(100 * r["test_macro_f1"], 2) for r in recs})
            rows.extend(recs)
    pd.DataFrame(rows).to_csv(paths.v3 / "detector_v4" / "summary.csv", index=False)


if __name__ == "__main__":
    main()
