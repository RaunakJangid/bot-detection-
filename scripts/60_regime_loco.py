"""Regime check, more settings: IoT-23 leave-one-capture-out (LOCO).

Row 28x2 rested on 11 settings with only 3 where ranking was lost. Here every IoT-23 capture with >= 100 benign and
>= 100 malicious flows (16 captures) is held out in turn: train on all other captures (cross-split de-dup, 15% random
validation slice of the training captures), test on the held-out capture. Models: XGBoost s0-2, RF s0, large MLP s0,
small MLP scratch s0-1, small MLP KD s0-1 (same recipes as scripts/41_iot23_shift.py). For every (capture, model):
the label-free predictors of scripts/51_regime_check.py (domain AUC, score KS, conformal abstention, ATC,
disagreement) and the ground-truth target ROC-AUC (Case 2 := AUC < 0.9). Then the predictors are re-evaluated on
all settings (row 28x2 instances + LOCO), at instance and setting level. FPR at the 1% validation target is stored too."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.data.datasets import stratified_sample
from shield.data.preprocess import dedup_across_splits, row_hash
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("regime_loco")


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


i23 = _load("41_iot23_shift.py")
reg = _load("51_regime_check.py")


def fit_small(X, y, Xv, yv, Xt, dev):
    S = {}
    it = stratified_sample(y, min(len(y), 2_000_000), np.random.default_rng(0), 1000)
    Xs, ys = X[it], y[it]
    for s in range(3):
        b = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                              early_stopping_rounds=50, random_state=s, subsample=0.8, colsample_bytree=0.8)
        b.fit(Xs, ys, eval_set=[(Xv, yv)], verbose=False)
        S[f"xgboost_s{s}"] = (b.predict_proba(Xv)[:, 1], b.predict_proba(Xt)[:, 1])
    rf = RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=0).fit(np.nan_to_num(Xs[:1_000_000]), ys[:1_000_000])
    S["rf_s0"] = (rf.predict_proba(np.nan_to_num(Xv))[:, 1], rf.predict_proba(np.nan_to_num(Xt))[:, 1])
    Xn, Xvn, Xtn = (np.nan_to_num(a).astype(np.float32) for a in (Xs, Xv, Xt))
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xn, Xvn, Xtn = ((a - mu) / sd for a in (Xn, Xvn, Xtn))
    big = i23.train_mlp(Xn, ys, Xvn, yv, (384, 384, 384, 384), 0, dev)
    S["large_mlp_s0"] = (i23.mlp_score(big, Xvn, dev), i23.mlp_score(big, Xtn, dev))
    for s in range(2):
        m = i23.train_mlp(Xn, ys, Xvn, yv, (64, 64), s, dev)
        S[f"small_mlp_scratch_s{s}"] = (i23.mlp_score(m, Xvn, dev), i23.mlp_score(m, Xtn, dev))
        m = i23.train_mlp(Xn, ys, Xvn, yv, (64, 64), s, dev, teacher=big)
        S[f"small_mlp_kd_s{s}"] = (i23.mlp_score(m, Xvn, dev), i23.mlp_score(m, Xtn, dev))
    return S


def evaluate(inst):
    preds = ["dom_auc", "score_ks", "conf_abst", "atc", "disagree"]
    case2 = np.array([r["target_auc"] < 0.9 for r in inst]); tauc = np.array([r["target_auc"] for r in inst])
    settings = sorted({r["setting"] for r in inst})
    st_ = np.array([np.median([r["target_auc"] for r in inst if r["setting"] == s]) for s in settings])
    out = {"n_instances": len(inst), "n_case2": int(case2.sum()), "n_settings": len(settings),
           "n_case2_settings": int((st_ < 0.9).sum()), "eval": {}}
    rng = np.random.default_rng(0)
    for p in preds:
        v = np.array([r[p] for r in inst])
        sv = np.array([np.median([r[p] for r in inst if r["setting"] == s]) for s in settings])
        boots = []
        for _ in range(1000):                      # bootstrap over settings for a CI on the setting-level AUC
            b = rng.integers(0, len(settings), len(settings))
            if 0 < (st_[b] < 0.9).sum() < len(b):
                boots.append(roc_auc_score(st_[b] < 0.9, sv[b]))
        out["eval"][p] = {"auc_case2_instances": float(roc_auc_score(case2, v)),
                          "auc_case2_settings": float(roc_auc_score(st_ < 0.9, sv)),
                          "auc_settings_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                          "spearman_settings": float(spearmanr(sv, st_).statistic)}
        log.info("%-10s case-2 AUC instances %.3f | settings %.3f [%.2f, %.2f] | Spearman %.2f", p,
                 out["eval"][p]["auc_case2_instances"], out["eval"][p]["auc_case2_settings"],
                 *out["eval"][p]["auc_settings_ci95"], out["eval"][p]["spearman_settings"])
    return out


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_f = P.v3 / "regime_loco.json"
    res = load_json(out_f) if out_f.exists() else {"instances": [], "done": []}
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    y = df.y.to_numpy()
    g = df.groupby("capture").y.agg(["size", "sum"])
    caps = [c for c in g.index if g.loc[c, "sum"] >= 100 and g.loc[c, "size"] - g.loc[c, "sum"] >= 100]
    log.info("LOCO captures: %d", len(caps))
    rng = np.random.default_rng(0)
    for c in caps:
        if c in res["done"]:
            continue
        is_test = (df.capture == c).to_numpy()
        tr_all = np.flatnonzero(~is_test)
        vmask = np.random.default_rng(1).random(len(tr_all)) < 0.15
        parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
        h = {s: row_hash(pd.DataFrame(np.nan_to_num(A[i], nan=-9.0))) for s, i in parts.items()}
        keep, _ = dedup_across_splits(h, {s: y[i] for s, i in parts.items()})
        parts = {s: i[keep[s]] for s, i in parts.items()}
        X, Xv, Xt = (A[parts[s]].astype(np.float32) for s in ("train", "validation", "test"))
        yy, yv, yt = (y[parts[s]] for s in ("train", "validation", "test"))
        if (yt == 0).sum() < 100 or (yt == 1).sum() < 100:
            log.info("%s: too few after de-dup, skipped", c); res["done"].append(c); continue
        S = fit_small(X, yy, Xv, yv, Xt, dev)
        dom = reg.dom_auc(np.nan_to_num(Xv), np.nan_to_num(Xt), rng)
        for m, (sv, st) in S.items():
            ref = S["large_mlp_s0"][1] if m.startswith(("xgb", "rf")) else S["xgboost_s0"][1]
            r = reg.predictors(sv, yv, st, ref, dom)
            thr = np.quantile(sv[yv == 0], 0.99)
            r.update({"setting": f"iot23_loco_{c}", "model": m, "target_auc": float(roc_auc_score(yt, st)),
                      "fpr_at_1pct": float((st[yt == 0] > thr).mean()), "recall_at_1pct": float((st[yt == 1] > thr).mean())})
            res["instances"].append(r)
        res["done"].append(c)
        log.info("%s: dom_auc %.3f | target AUC %s", c, dom,
                 {m: round(r["target_auc"], 3) for r in res["instances"] if r["setting"] == f"iot23_loco_{c}" for m in [r["model"]]})
        save_json(res, out_f)
    old = load_json(P.v3 / "regime_check.json")["instances"]
    log.info("== LOCO only"); res["eval_loco"] = evaluate(res["instances"])
    log.info("== all settings (row 28x2 + LOCO)"); res["eval_all"] = evaluate(old + res["instances"])
    save_json(res, out_f)


if __name__ == "__main__":
    main()
