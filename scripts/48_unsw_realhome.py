"""E7 (Zeek route): false alarms on real smart-home benign traffic (UNSW IoT traces, 3 days, 28 IoT devices plus phones
and laptops), in the same Zeek flow-feature space as IoT-23.

Version check (done before this run, scratchpad zeek_compare.py): Zeek 8.0.10 on IoT-23's own honeypot pcaps reproduces
IoT-23's conn.log rows 1:1 (452/452; 1357/1374 on a truncated copy); duration, bytes and conn_state 100% identical,
packet / IP-byte counts 98.5-100%, history 96-98%, service only 77-91% (Zeek 8 detects more services). The
`service` one-hot features are therefore dropped for this experiment.
Protocol: detectors trained on ALL IoT-23 captures (de-duplicated; 15% random validation slice for early stopping and
the 1% validation-benign threshold). UNSW flows are benign by construction (no attacks in the traces), so recall
cannot be measured there; reported: realised FPR and false alarms per day at the fixed threshold, and after a
time-ordered commissioning recalibration (conformal 1% quantile of the first 1000 flows of day 1), plus the lab
recall (IoT-23 validation attacks) at each threshold, and a benign domain AUC (IoT-23 validation benign vs UNSW)."""

import glob
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.data.preprocess import dedup_mask, row_hash
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("unsw")
_s = importlib.util.spec_from_file_location("i23", __file__.replace("48_unsw_realhome", "41_iot23_shift"))
i23 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i23)


def read_zeek(p):
    with open(p) as f:
        cols = next(l for l in f if l.startswith("#fields")).rstrip("\n").split("\t")[1:]
    z = pd.read_csv(p, sep="\t", comment="#", names=cols, dtype=str)
    z["ts"] = z["ts"].astype(float)
    return z


def conformal(s, a=0.01):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    torch.set_num_threads(8); dev = torch.device("cpu")
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    keep_cols = [j for j, n in enumerate(names) if not n.startswith("svc_")]
    A = A[:, keep_cols]; names = [names[j] for j in keep_cols]
    y = df.y.to_numpy()
    k, _ = dedup_mask(row_hash(pd.DataFrame(np.nan_to_num(A, nan=-9.0))), y)
    A, y = np.nan_to_num(A[k]).astype(np.float32), y[k]
    vm = np.random.default_rng(7).random(len(y)) < 0.15
    X, yy, Xv, yv = A[~vm], y[~vm], A[vm], y[vm]
    days = sorted(glob.glob(str(P.v3 / "realbenign" / "zeek_unsw" / "conn_*.log")))
    U = {os.path.basename(f)[5:-4]: read_zeek(f) for f in days}
    Ux = {d: np.nan_to_num(i23.featurize(z)[0][:, keep_cols]).astype(np.float32) for d, z in U.items()}
    allU = np.vstack(list(Ux.values()))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    S = {}
    for s in range(3):
        b = xgb.XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=8,
                              early_stopping_rounds=50, random_state=s, subsample=0.8, colsample_bytree=0.8)
        b.fit(X, yy, eval_set=[(Xv, yv)], verbose=False)
        S[f"xgb_s{s}"] = lambda Z, b=b: b.predict_proba(Z)[:, 1]
    big = i23.train_mlp((X - mu) / sd, yy, (Xv - mu) / sd, yv, (384, 384, 384, 384), 0, dev, epochs=6)
    S["large_mlp_s0"] = lambda Z: i23.mlp_score(big, (Z - mu) / sd, dev)
    for s in range(3):
        m = i23.train_mlp((X - mu) / sd, yy, (Xv - mu) / sd, yv, (64, 64), s, dev, epochs=6)
        S[f"small_s{s}"] = lambda Z, m=m: i23.mlp_score(m, (Z - mu) / sd, dev)
        m2 = i23.train_mlp((X - mu) / sd, yy, (Xv - mu) / sd, yv, (64, 64), s, dev, teacher=big, epochs=6)
        S[f"small_kd_s{s}"] = lambda Z, m=m2: i23.mlp_score(m, (Z - mu) / sd, dev)
    res = {"features": names, "days": {d: {"flows": int(len(z)), "hours": float((z.ts.max() - z.ts.min()) / 3600)}
                                       for d, z in U.items()}, "models": {}}
    d1 = list(U)[0]
    order1 = np.argsort(U[d1].ts.to_numpy(), kind="stable")
    for name, fn in S.items():
        sv = fn(Xv); t_fix = float(np.quantile(sv[yv == 0], 0.99))
        s1 = fn(Ux[d1])[order1]; t_com = float(conformal(s1[:1000]))
        r = {"lab_val_recall_fixed": float((sv[yv == 1] > t_fix).mean()),
             "lab_val_recall_commission": float((sv[yv == 1] > t_com).mean()), "days": {}}
        for d, Z in Ux.items():
            sc = fn(Z); hrs = res["days"][d]["hours"]
            ev = np.ones(len(sc), bool)
            if d == d1:
                ev[order1[:1000]] = False
            fpr_f, fpr_c = float((sc > t_fix).mean()), float((sc[ev] > t_com).mean())
            r["days"][d] = {"fpr_fixed": fpr_f, "alarms_per_day_fixed": fpr_f * len(sc) / max(hrs, 1e-9) * 24,
                            "fpr_commission": fpr_c, "alarms_per_day_commission": fpr_c * len(sc) / max(hrs, 1e-9) * 24}
        res["models"][name] = r
        log.info("%s: lab recall fixed %.3f / commission %.3f | %s", name, r["lab_val_recall_fixed"], r["lab_val_recall_commission"],
                 {d: (round(100 * v["fpr_fixed"], 2), round(100 * v["fpr_commission"], 2), int(v["alarms_per_day_fixed"]),
                      int(v["alarms_per_day_commission"])) for d, v in r["days"].items()})
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    B = Xv[yv == 0]; kk = min(len(B), len(allU), 50_000)
    M = np.vstack([B[rng.choice(len(B), kk, replace=False)], allU[rng.choice(len(allU), kk, replace=False)]])
    lab = np.r_[np.zeros(kk), np.ones(kk)]; fold = rng.integers(0, 2, 2 * kk); sc = np.zeros(2 * kk)
    for f in (0, 1):
        sc[fold == f] = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(M[fold != f], lab[fold != f]).predict_proba(M[fold == f])[:, 1]
    res["domain_auc_iot23_vs_unsw_benign"] = float(roc_auc_score(lab, sc))
    log.info("days %s | domain AUC %.3f", res["days"], res["domain_auc_iot23_vs_unsw_benign"])
    save_json(res, P.v3 / "realbenign" / "unsw_realhome.json")


if __name__ == "__main__":
    main()
