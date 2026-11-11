"""v4: dump real-home (UNSW) and IoT-23 validation scores for the models of scripts/48_unsw_realhome.py.

Same data, de-duplication, validation slice, model recipes and seeds as 48 (row 28ae); 48 stored only summaries.
Saved to outputs_v4/regime/unsw.npz: models, sv [M, nv] and yv (IoT-23 validation), and per day d: u_<d> [M, n_d]
scores in time order with ts_<d>. Used for the time-ordered benign null test of the KS warning and for commissioning
contamination with injected IoT-23 attack flows. The per-day FPR at the fixed and commissioned thresholds is
recomputed and logged, so agreement with unsw_realhome.json can be checked (MLPs are retrained on CPU)."""

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from shield.cli import parser
from shield.data.preprocess import dedup_mask, row_hash
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("unsw_v4_dump")
_s = importlib.util.spec_from_file_location("u48", Path(__file__).with_name("48_unsw_realhome.py"))
u48 = importlib.util.module_from_spec(_s); _s.loader.exec_module(u48)
i23 = u48.i23


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    out = Path(str(P.v3).replace("outputs_v3", "outputs_v4")) / "regime"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); dev = torch.device("cpu")
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    keep_cols = [j for j, n in enumerate(names) if not n.startswith("svc_")]
    A = A[:, keep_cols]
    y = df.y.to_numpy()
    k, _ = dedup_mask(row_hash(pd.DataFrame(np.nan_to_num(A, nan=-9.0))), y)
    A, y = np.nan_to_num(A[k]).astype(np.float32), y[k]
    vm = np.random.default_rng(7).random(len(y)) < 0.15
    X, yy, Xv, yv = A[~vm], y[~vm], A[vm], y[vm]
    days = sorted(glob.glob(str(P.v3 / "realbenign" / "zeek_unsw" / "conn_*.log")))
    U = {os.path.basename(f)[5:-4]: u48.read_zeek(f) for f in days}
    order = {d: np.argsort(z.ts.to_numpy(), kind="stable") for d, z in U.items()}
    Ux = {d: np.nan_to_num(i23.featurize(z)[0][:, keep_cols]).astype(np.float32)[order[d]] for d, z in U.items()}
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
    models = list(S)
    arr = {"models": np.array(models), "yv": yv.astype(np.int8),
           "sv": np.stack([S[m](Xv) for m in models]).astype(np.float32), "days": np.array(list(U))}
    for d in U:
        arr[f"u_{d}"] = np.stack([S[m](Ux[d]) for m in models]).astype(np.float32)
        arr[f"ts_{d}"] = U[d].ts.to_numpy()[order[d]]
    np.savez_compressed(out / "unsw.npz", **arr)
    d1 = list(U)[0]
    for i, m in enumerate(models):
        sv = arr["sv"][i]; t_fix = np.quantile(sv[yv == 0], 0.99); t_com = u48.conformal(arr[f"u_{d1}"][i][:1000])
        log.info("%s: fixed %s | commissioned %s", m, [round(100 * float((arr[f"u_{d}"][i] > t_fix).mean()), 2) for d in U],
                 [round(100 * float((arr[f"u_{d}"][i][1000 if d == d1 else 0:] > t_com).mean()), 2) for d in U])


if __name__ == "__main__":
    main()
