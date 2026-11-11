"""v4 regime analyses, step 1: dump per-instance validation and deployment scores for all 18 regime-check settings.

Rows 28x2 / 28aj (scripts/51_regime_check.py, 60_regime_loco.py) stored only summary predictors per (setting, model).
The v4 analyses (recall at the target FPR, fixed-window KS, held-out warning threshold, attack-share sweep,
commissioning contamination) need the scores themselves. This script repeats the sampling of 51 (same rng order, so
the same 200k-row subsamples) and retrains the LOCO models of 60 (same recipes and seeds), then saves, per setting,
outputs_v4/regime/<setting>.npz with models, sv [M, nv], yv, st [M, nt], yt, ts (deployment flow time, NaN if none),
cap (IoT-23 capture of each deployment flow, '' if none) and the target ROC-AUC recomputed per model, so agreement
with the stored v3 instances can be checked (LOCO MLPs are retrained, so small differences are expected there)."""

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.data.preprocess import dedup_across_splits, row_hash
from shield.eval.metrics import softmax
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("regime_v4_dump")


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


reg = _load("51_regime_check.py")
loco = _load("60_regime_loco.py")
i23 = loco.i23


def save(out, setting, S, yv, yt, ts=None, cap=None):
    models = list(S)
    sv = np.stack([S[m][0] for m in models]).astype(np.float32)
    st = np.stack([S[m][1] for m in models]).astype(np.float32)
    auc = np.array([roc_auc_score(yt, s) for s in st])
    np.savez_compressed(out / f"{setting}.npz", models=np.array(models), sv=sv, yv=np.asarray(yv, np.int8), st=st,
                        yt=np.asarray(yt, np.int8), ts=np.full(len(yt), np.nan) if ts is None else np.asarray(ts, float),
                        cap=np.array([""] * len(yt)) if cap is None else np.asarray(cap).astype(str), target_auc=auc)
    log.info("%s: %d models, nv %d, nt %d, median target AUC %.4f", setting, len(models), sv.shape[1], st.shape[1],
             np.median(auc))


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    out = Path(str(P.v3).replace("outputs_v3", "outputs_v4")) / "regime"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # --- CICIoMT (shift) and CICIoT (control): same rng order as 51
    for ds, task, setting in (("ciciomt", "binary", "ciciomt_binary"), ("ciciomt", "category", "ciciomt_category"),
                              ("ciciomt", "attack", "ciciomt_attack"), ("ciciot", "class34", "ciciot_control")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        ben = next(i for i, c in enumerate(d.classes) if c in ("Benign", "BenignTraffic"))
        yv = (d.splits["validation"].y != ben).astype(int); yt = (d.splits["test"].y != ben).astype(int)
        iv = rng.choice(len(yv), min(len(yv), 200_000), replace=False); it = rng.choice(len(yt), min(len(yt), 200_000), replace=False)
        Xv = np.asarray(d.splits["validation"].X)[iv]; Xt = np.asarray(d.splits["test"].X)[it]
        reg.dom_auc(Xv, Xt, rng)
        lc = P.v3 / "logits" / f"{ds}_{task}"
        models = sorted(os.path.basename(f)[:-len("_validation.npy")] for f in glob.glob(str(lc / "*_validation.npy"))
                        if os.path.exists(str(f).replace("_validation", "_test")))
        S = {m: (1 - softmax(np.load(lc / f"{m}_validation.npy").astype(np.float64)[iv])[:, ben],
                 1 - softmax(np.load(lc / f"{m}_test.npy").astype(np.float64)[it])[:, ben]) for m in models}
        save(out, setting, S, yv[iv], yt[it])
    # --- cross-dataset, same recipe and rng order as 51
    import xgboost as xgb
    from sklearn.neural_network import MLPClassifier
    xd = {k: load_processed(P.processed(k), "binary") for k in ("xciciot", "xciciomt")}
    for src, dst in (("xciciot", "xciciomt"), ("xciciomt", "xciciot")):
        a, b = xd[src], xd[dst]
        tr = a.splits["train"]; it = stratified_sample(tr.y, 1_000_000, rng, 1000)
        X, y = np.asarray(tr.X[it], np.float32), tr.y[it]
        Xv = np.asarray(a.splits["validation"].X, np.float32); yv_ = a.splits["validation"].y
        Xt = np.asarray(b.splits["test"].X, np.float32); yt_ = b.splits["test"].y
        ben_a = a.classes.index("Benign"); ben_b = b.classes.index("Benign")
        yvb, ytb = (yv_ != ben_a).astype(int), (yt_ != ben_b).astype(int)
        iv = rng.choice(len(yvb), min(len(yvb), 200_000), replace=False); itt = rng.choice(len(ytb), min(len(ytb), 200_000), replace=False)
        reg.dom_auc(Xv[iv], Xt[itt], rng)
        S = {}
        for s in range(3):
            c = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=8,
                                  random_state=s, subsample=0.8, colsample_bytree=0.8).fit(X, (y != ben_a).astype(int))
            S[f"xgb_s{s}"] = (c.predict_proba(Xv[iv])[:, 1], c.predict_proba(Xt[itt])[:, 1])
        for s in range(2):
            m = MLPClassifier((64, 64), max_iter=30, random_state=s).fit(X[:300_000], (y[:300_000] != ben_a).astype(int))
            S[f"mlp_s{s}"] = (m.predict_proba(Xv[iv])[:, 1], m.predict_proba(Xt[itt])[:, 1])
        save(out, f"cross_{src}_to_{dst}", S, yvb[iv], ytb[itt])
    # --- IoT-23 designs from the cached scores (within capture), with flow times
    for design in ("same_family", "capture_shift", "random_control"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True)
        models, yt, cap = list(Z["models"]), Z["yt"], Z["capture"]
        groups = np.unique(cap) if design != "random_control" else ["all"]
        for c in groups:
            m_ = cap == c if design != "random_control" else np.ones(len(yt), bool)
            if (yt[m_] == 0).sum() < 100 or (yt[m_] == 1).sum() < 100:
                continue
            S = {m: (Z["sv"][i], Z["st"][i][m_]) for i, m in enumerate(models)}
            save(out, f"iot23_{design}_{c}", S, Z["yv"], yt[m_], Z["ts"][m_], cap[m_])
    # --- IoT-23 LOCO: retrain with the recipes of 60
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, _ = i23.featurize(df)
    y = df.y.to_numpy(); ts_all = df.ts.to_numpy(float); cap_all = df.capture.to_numpy()
    g = df.groupby("capture").y.agg(["size", "sum"])
    caps = [c for c in g.index if g.loc[c, "sum"] >= 100 and g.loc[c, "size"] - g.loc[c, "sum"] >= 100]
    for c in caps:
        if (out / f"iot23_loco_{c}.npz").exists():
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
            log.info("%s: too few after de-dup, skipped", c); continue
        S = loco.fit_small(X, yy, Xv, yv, Xt, dev)
        save(out, f"iot23_loco_{c}", S, yv, yt, ts_all[parts["test"]], cap_all[parts["test"]])


if __name__ == "__main__":
    main()
