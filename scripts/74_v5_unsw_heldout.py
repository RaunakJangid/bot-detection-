"""v5 B2: real-home models with a held-out recall. v4 (scripts/68) trained the 10 real-home models on a random split of
the pooled IoT-23 sample, so their "lab recall" was measured on validation flows of the training captures.

Here the same 10 recipes (XGBoost s0-2 with early stopping, large MLP 384x4, small MLP 64-64 s0-2 scratch and distilled,
6 epochs; Zeek features without service one-hots; exact duplicates removed across splits) are trained on the TRAINING
captures of two IoT-23 designs of scripts/41 (same capture assignment and seeds):
  same_family   held-out captures contain malware families that remain in training
  capture_shift held-out captures are a random third of the malware captures (families may be unseen)
Validation = 15% random slice of the training captures (lab threshold t0, early stopping). Saved per design to
outputs_v5/regime/unsw_<design>.npz: the v4 real-home keys (models, sv, yv, days, u_<d>, ts_<d>) plus held-out
attack scores ha [M, n] and their capture ha_cap, held-out benign scores hb [M, n] and hb_cap.
Usage: 74_v5_unsw_heldout.py"""

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from shield.data.preprocess import dedup_across_splits, row_hash
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("unsw_v5_heldout")
_s = importlib.util.spec_from_file_location("u48", Path(__file__).with_name("48_unsw_realhome.py"))
u48 = importlib.util.module_from_spec(_s); _s.loader.exec_module(u48)
i23 = u48.i23


def designs(caps):
    """Capture assignment of scripts/41 (same seeds and rules)."""
    mal = [c for c in caps if "Malware" in c]
    order = np.random.default_rng(20261011).permutation(mal)
    test_caps = set(order[: len(mal) // 3]) | {c for c in caps if c.startswith("CTU-Honeypot-Capture-7")}
    fam_of = lambda c: i23.FAMILY[c.split("Capture-")[1]]
    r3 = np.random.default_rng(20261012)
    same_caps = {c for c in caps if c.startswith("CTU-Honeypot-Capture-7")}
    for f in sorted({fam_of(c) for c in mal}):
        cs = sorted(c for c in mal if fam_of(c) == f)
        if len(cs) >= 2:
            same_caps |= set(r3.permutation(cs)[: max(1, len(cs) // 3)])
    return {"same_family": same_caps, "capture_shift": test_caps}


def main():
    P = Paths(False); torch.set_num_threads(2)
    out = Path("D:/shield_run/outputs_v5/regime"); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    keep_cols = [j for j, n in enumerate(names) if not n.startswith("svc_")]
    A = np.nan_to_num(A[:, keep_cols]).astype(np.float32)
    y = df.y.to_numpy(); cap = df.capture.to_numpy()
    days = sorted(glob.glob(str(P.v3 / "realbenign" / "zeek_unsw" / "conn_*.log")))
    U = {os.path.basename(f)[5:-4]: u48.read_zeek(f) for f in days}
    order = {d: np.argsort(z.ts.to_numpy(), kind="stable") for d, z in U.items()}
    Ux = {d: np.nan_to_num(i23.featurize(z)[0][:, keep_cols]).astype(np.float32)[order[d]] for d, z in U.items()}
    for design, test_caps in designs(sorted(df.capture.unique())).items():
        if (out / f"unsw_{design}.npz").exists():
            continue
        is_test = np.isin(cap, list(test_caps))
        tr_all = np.flatnonzero(~is_test)
        vmask = np.random.default_rng(1).random(len(tr_all)) < 0.15
        parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
        h = {s: row_hash(pd.DataFrame(A[i])) for s, i in parts.items()}
        keep, st = dedup_across_splits(h, {s: y[i] for s, i in parts.items()})
        parts = {s: i[keep[s]] for s, i in parts.items()}
        X, yy = A[parts["train"]], y[parts["train"]]
        Xv, yv = A[parts["validation"]], y[parts["validation"]]
        Xt, yt, ct = A[parts["test"]], y[parts["test"]], cap[parts["test"]]
        log.info("%s: train %d (mal %.3f) val %d test %d (mal %d, benign %d); test captures %s", design, len(yy), yy.mean(),
                 len(yv), len(yt), int(yt.sum()), int((yt == 0).sum()), sorted(test_caps))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        S = {}
        for s in range(3):
            b = xgb.XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=4,
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
        models = list(S)
        arr = {"models": np.array(models), "yv": yv.astype(np.int8), "days": np.array(list(U)),
               "sv": np.stack([S[m](Xv) for m in models]).astype(np.float32),
               "ha": np.stack([S[m](Xt[yt == 1]) for m in models]).astype(np.float32), "ha_cap": ct[yt == 1].astype(str),
               "hb": np.stack([S[m](Xt[yt == 0]) for m in models]).astype(np.float32), "hb_cap": ct[yt == 0].astype(str),
               "test_caps": np.array(sorted(test_caps))}
        for d in U:
            arr[f"u_{d}"] = np.stack([S[m](Ux[d]) for m in models]).astype(np.float32)
            arr[f"ts_{d}"] = U[d].ts.to_numpy()[order[d]]
        np.savez_compressed(out / f"unsw_{design}.npz", **arr)
        for i, m in enumerate(models):
            t0 = np.quantile(arr["sv"][i][yv == 0], 0.99)
            log.info("%s %s: val recall %.3f | held-out attack recall %.3f | held-out benign FPR %.4f | home FPR %s", design, m,
                     float((arr["sv"][i][yv == 1] > t0).mean()), float((arr["ha"][i] > t0).mean()),
                     float((arr["hb"][i] > t0).mean()), [round(100 * float((arr[f"u_{d}"][i] > t0).mean()), 2) for d in U])


if __name__ == "__main__":
    main()
