"""P2 v4, experiment X1: per-cascade detector cost, every cascade used in P2. Output: outputs_v5/p2/throughput_v4.json.

One pinned CPU core, one thread, batch 1 and batch 256, forward pass plus the softmax/entropy needed for the escalation
decision; median of repeated timings on real test rows (not random inputs).
  CIC (ciciot_class34, ciciomt_binary/category/attack): student shield_s0 in fp32 (torch), the same student as the
    integer-only IntMLP of scripts/13 and 65 (numpy int8 weights / int32 accumulators), and the residual-MLP teacher_s0.
  IoT-23 (capture-shift design of scripts/41, same split and de-duplication): the small distilled MLP (64-64, KD from a
    384x4 MLP, seed 0), and the ensemble expert = mean of XGBoost s0 (2,000 trees, depth 10, early stopping) and the
    384x4 MLP s0, retrained with the recipes of scripts/41 (cost only; the escalation results of P2 use the v3 scores).
Run when no other CPU-heavy job is active (timings)."""

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from shield.data.datasets import load_processed, stratified_sample
from shield.data.preprocess import dedup_across_splits, row_hash
from shield.eval.latency import _PinnedCore
from shield.eval.predictions import load_student, run_dir
from shield.models.quant import IntMLP, calibrate
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json

torch.set_num_threads(1)
OUT = Path("D:/shield_run/outputs_v5/p2"); OUT.mkdir(parents=True, exist_ok=True)
REPS = {1: 3000, 256: 300}


def timeit(fn, X, b):
    n = len(X); idx = np.random.default_rng(0).integers(0, n - b, REPS[b] + 50)
    for s in idx[:50]:
        fn(X[s:s + b])
    t = np.empty(REPS[b])
    for i, s in enumerate(idx[50:]):
        t0 = time.perf_counter(); fn(X[s:s + b]); t[i] = time.perf_counter() - t0
    med = float(np.median(t))
    return {"latency_ms": 1e3 * med, "p95_ms": 1e3 * float(np.quantile(t, .95)), "throughput": b / med}


def ent(logits):
    z = logits - logits.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def balanced_sample(y, per_class, rng):
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
                                   for c in np.unique(y)]))


def main():
    P = Paths(False); tcfg = load_config("teacher", False); out = {}
    with _PinnedCore(0):
        for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
            data = load_processed(P.processed(ds), task); te, tr = data.splits["test"], data.splits["train"]
            model, S = load_student(run_dir(P, ds, task, "shield", 0, tcfg["model"]), "cpu")
            cols = lambda X: X if len(S) == X.shape[1] else np.asarray(X[:, S])
            Xs = np.ascontiguousarray(cols(np.asarray(te.X[:200000])), dtype=np.float32)
            Xf = np.ascontiguousarray(np.asarray(te.X[:200000]), dtype=np.float32)
            rec = load_json(P.v3 / "int8" / f"{ds}_{task}" / "shield_s0.json")["row"]
            cset, ip, ap = rec["calibration_w8a8"].split("/")
            rng = np.random.default_rng(0)
            prop = cols(np.asarray(tr.X[stratified_sample(tr.y, 200_000, rng, 50)]))
            cal = prop if cset == "prop" else cols(np.asarray(tr.X[balanced_sample(tr.y, 200_000 // data.n_classes, rng)]))
            q = IntMLP(model, calibrate(model, cal, float(ip), float(ap)), act_bits=8)
            teacher, ck = load_teacher(teacher_dir(P, ds, task, tcfg["model"], 0), torch.device("cpu"))
            with torch.inference_mode():
                f_s = lambda x: ent(model(torch.from_numpy(x)).numpy())
                f_t = lambda x: teacher(torch.from_numpy(x)).numpy().argmax(1)
                f_q = lambda x: ent(q.forward(x))
                out[f"{ds}_{task}"] = {b: {"student_fp32": timeit(f_s, Xs, b), "student_int8": timeit(f_q, Xs, b),
                                           "teacher": timeit(f_t, Xf, b)} for b in (1, 256)}
                out[f"{ds}_{task}"]["params"] = {"student": int(sum(p.numel() for p in model.parameters())),
                                                 "teacher": int(sum(p.numel() for p in teacher.parameters()))}
            print(ds, task, json.dumps(out[f"{ds}_{task}"])[:600], flush=True)
    json.dump(out, open(OUT / "throughput_v4.json", "w"), indent=1)         # CIC part saved first
    # IoT-23 cascade (capture-shift design), retrained for cost
    _s = importlib.util.spec_from_file_location("i23", Path(__file__).with_name("41_iot23_shift.py"))
    i23 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i23)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet"); A, _ = i23.featurize(df); y = df.y.to_numpy()
    mal = sorted(c for c in df.capture.unique() if "Malware" in c)
    order = np.random.default_rng(20261011).permutation(mal)
    test_caps = set(order[: len(mal) // 3]) | {c for c in df.capture.unique() if c.startswith("CTU-Honeypot-Capture-7")}
    is_test = df.capture.isin(test_caps).to_numpy(); tr_all = np.flatnonzero(~is_test)
    vmask = np.random.default_rng(1).random(len(tr_all)) < 0.15
    parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
    h = {s: row_hash(pd.DataFrame(np.nan_to_num(A[i], nan=-9.0))) for s, i in parts.items()}
    keep, _ = dedup_across_splits(h, {s: y[i] for s, i in parts.items()}); parts = {s: i[keep[s]] for s, i in parts.items()}
    X, yy, Xv, yv, Xt = A[parts["train"]], y[parts["train"]], A[parts["validation"]], y[parts["validation"]], A[parts["test"]]
    it = stratified_sample(yy, min(len(yy), 2_000_000), np.random.default_rng(0), 1000); Xs_, ys_ = X[it], yy[it]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                          early_stopping_rounds=50, random_state=0, subsample=0.8, colsample_bytree=0.8)
    b.fit(Xs_, ys_, eval_set=[(Xv, yv)], verbose=False)
    Xn = np.nan_to_num(Xs_).astype(np.float32); mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xvn = ((np.nan_to_num(Xv) - mu) / sd).astype(np.float32); ys_ = np.asarray(ys_); yv = np.asarray(yv)
    big = i23.train_mlp(((Xn - mu) / sd).astype(np.float32), ys_, Xvn, yv, (384, 384, 384, 384), 0, dev)
    small = i23.train_mlp(((Xn - mu) / sd).astype(np.float32), ys_, Xvn, yv, (64, 64), 0, dev, teacher=big)
    big, small = big.cpu().eval(), small.cpu().eval()
    booster = b.get_booster(); booster.set_param({"device": "cpu", "nthread": 1})
    Xtt = np.ascontiguousarray(((np.nan_to_num(Xt) - mu) / sd), dtype=np.float32); Xraw = np.ascontiguousarray(Xt, dtype=np.float32)
    with _PinnedCore(0), torch.inference_mode():
        f_small = lambda x: ent(small(torch.from_numpy(x)).numpy())
        f_big = lambda x: big(torch.from_numpy(x)).numpy()
        f_xgb = lambda x: booster.inplace_predict(x)
        out["iot23_capture_shift"] = {bb: {"student_fp32": timeit(f_small, Xtt, bb), "mlp_large": timeit(f_big, Xtt, bb),
                                           "xgboost": timeit(f_xgb, Xraw, bb)} for bb in (1, 256)}
    for bb in (1, 256):
        r = out["iot23_capture_shift"][bb]
        r["ensemble"] = {"throughput": 1.0 / (1.0 / r["mlp_large"]["throughput"] + 1.0 / r["xgboost"]["throughput"])}
    out["iot23_capture_shift"]["xgboost_trees"] = int(b.best_iteration + 1)
    print("iot23", json.dumps(out["iot23_capture_shift"])[:800], flush=True)
    json.dump(out, open(OUT / "throughput_v4.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
