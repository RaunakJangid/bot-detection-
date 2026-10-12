"""E4: how much does the evaluation protocol inflate published-style scores? Factorial ablation.

Factors (each switched alone and together, same model and settings in every cell):
  IAT      the leaky IAT feature kept vs removed
  dedup    exact-duplicate / label-conflict removal across train -> validation -> test, on vs off
  split    CICIoMT only: official capture-level split (train captures vs new test captures) vs a random
           row split of the pooled data (what most papers do); CICIoT2023 ships no capture IDs, so its split
           is random in every cell
Models: XGBoost (Almahaqeri et al. 2026 style; 1 seed) and Random Forest (Neto et al. 2023 baseline;
1 seed). Trees need no scaling. Training rows capped at --cap (stratified, min 1000 per class), early stopping on
validation, test scored once per cell. Reported: test macro-F1 and accuracy (the literature's metric)."""

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from shield.cli import parser
from shield.data.datasets import stratified_sample
from shield.data.preprocess import (LABEL, cic_task_specs, dedup_across_splits, load_cic_frames, row_hash,
                                    stratified_split)
from shield.eval.metrics import classification_metrics
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("ablation")
JOBS = [("ciciot", "class34"), ("ciciomt", "category"), ("ciciomt", "attack")]


def frames_for(src, P, cfg, split):
    rng = np.random.default_rng(cfg["seed"])
    fr = load_cic_frames(src, P.interim(src), cfg, rng)
    if split == "official":
        return fr
    pool = pd.concat([fr["train"], fr["validation"], fr["test"]], ignore_index=True)
    parts = stratified_split(pd.factorize(pool[LABEL])[0], (0.7, 0.15, 0.15), np.random.default_rng(cfg["seed"]))
    return {s: pool.iloc[parts[s]].reset_index(drop=True) for s in ("train", "validation", "test")}


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--cap", type=int, default=2_000_000)
    args = p.parse_args()
    P, cfg = Paths(args.smoke), load_config("data", args.smoke)
    out_f = P.v3 / "ablation" / "protocol_ablation.json"
    res = load_json(out_f) if out_f.exists() else {}   # resumable
    for src in ("ciciot", "ciciomt"):
        splits = ("official", "random") if src == "ciciomt" else ("official",)
        for split in splits:
            fr = frames_for(src, P, cfg, split)
            labels = sorted(set().union(*(set(df[LABEL].unique()) for df in fr.values())))
            specs = cic_task_specs(src, labels, False)
            all_feats = [c for c in fr["train"].columns if c != LABEL]
            for iat, dedup in itertools.product((False, True), (True, False)):
                feats = [c for c in all_feats if iat or c != "IAT"]
                lab_index = {l: i for i, l in enumerate(labels)}
                codes = {s: df[LABEL].map(lab_index).to_numpy(np.int64) for s, df in fr.items()}
                if dedup:
                    keep, _ = dedup_across_splits({s: row_hash(df[feats]) for s, df in fr.items()}, codes)
                else:
                    keep = {s: np.ones(len(df), bool) for s, df in fr.items()}
                X = {s: fr[s][feats].to_numpy(np.float32)[keep[s]] for s in fr}
                for s in X:
                    X[s][~np.isfinite(X[s])] = np.nan
                for ds_, task in [j for j in JOBS if j[0] == src]:
                    classes, fn = specs[task]
                    ci = {c: i for i, c in enumerate(classes)}
                    lut = np.array([ci[fn(l)] for l in labels])
                    y = {s: lut[codes[s][keep[s]]] for s in fr}
                    n = len(classes)
                    for model, seeds in (("xgboost", (0,)), ("rf", (0,))):
                        for seed in seeds:
                            key = f"{src}_{task}|split={split}|IAT={'kept' if iat else 'removed'}|dedup={'on' if dedup else 'off'}|{model}|s{seed}"
                            if key in res:
                                continue
                            rng = np.random.default_rng(seed)
                            cap = args.cap if model == "xgboost" else args.cap // 2
                            it = stratified_sample(y["train"], min(cap, len(y["train"])), rng, 1000)
                            iv = stratified_sample(y["validation"], min(500_000, len(y["validation"])), rng, 200)
                            present = np.unique(y["train"][it]); rm = np.full(n, -1); rm[present] = np.arange(len(present))
                            iv = iv[rm[y["validation"][iv]] >= 0]
                            if model == "xgboost":
                                clf = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1,
                                                        tree_method="hist", device="cuda", random_state=seed,
                                                        early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.8)
                                clf.fit(X["train"][it], rm[y["train"][it]],
                                        eval_set=[(X["validation"][iv], rm[y["validation"][iv]])], verbose=False)
                            else:
                                clf = RandomForestClassifier(n_estimators=100, n_jobs=12, random_state=seed)
                                clf.fit(np.nan_to_num(X["train"][it]), rm[y["train"][it]])
                            Xt = X["test"] if model == "xgboost" else np.nan_to_num(X["test"])
                            pr = np.zeros((len(Xt), n))
                            for i in range(0, len(Xt), 500_000):
                                pr[i:i + 500_000][:, present] = clf.predict_proba(Xt[i:i + 500_000])
                            m = classification_metrics(y["test"], np.log(np.clip(pr, 1e-12, 1)), n)
                            res[key] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                                        "n_train": int(len(it)), "n_test": int(len(y["test"]))}
                            log.info("%s: macro-F1 %.4f acc %.4f (test rows %d)", key, m["macro_f1"], m["accuracy"],
                                     len(y["test"]))
                            save_json(res, out_f)


if __name__ == "__main__":
    main()
