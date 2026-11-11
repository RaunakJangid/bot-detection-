"""P2 v4, T1b: end-to-end capacity curve of the IoT-23 cascade (extends scripts/98 to the second cascade family).

Phase --train (GPU): the IoT-23 cost models of scripts/77 (capture-shift design of scripts/41, same split, the same
de-duplication and recipes): distilled student MLP 64-64, large MLP 384x4, XGBoost (2,000 trees max, depth 10, early
stopping); the expert is the ensemble (mean probability of XGBoost and the large MLP). Saved to
outputs_v5/p2/iot23_cost_models/ with the test rows (standardised and raw).
Phase --time (CPU, pinned core 0, one thread; run when the CPU is otherwise quiet): as scripts/98: separate student and
ensemble costs at batch 256 and 1; end-to-end pipeline at e in {0, 2, 5, 10, 15, 20}% (escalate the flows whose student
entropy exceeds the stream's (1 - e) quantile): batch-256 throughput, batch-1 per-flow latency p50/p95/p99 and
throughput; formula 1 / (1/mu_s + e/mu_t). Output: outputs_v5/p2/capacity_curve_iot23.json."""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2"); MD = OUT / "iot23_cost_models"
ES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)


def ent2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def train():
    import pandas as pd
    import torch
    import xgboost as xgb
    from shield.data.datasets import stratified_sample
    from shield.data.preprocess import dedup_across_splits, row_hash
    from shield.utils.config import Paths
    P = Paths(False); MD.mkdir(parents=True, exist_ok=True)
    s = importlib.util.spec_from_file_location("i23", Path(__file__).with_name("41_iot23_shift.py"))
    i23 = importlib.util.module_from_spec(s); s.loader.exec_module(i23)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet"); A, _ = i23.featurize(df); y = df.y.to_numpy()
    mal = sorted(c for c in df.capture.unique() if "Malware" in c)
    order = np.random.default_rng(20261011).permutation(mal)
    test_caps = set(order[: len(mal) // 3]) | {c for c in df.capture.unique() if c.startswith("CTU-Honeypot-Capture-7")}
    is_test = df.capture.isin(test_caps).to_numpy(); tr_all = np.flatnonzero(~is_test)
    vmask = np.random.default_rng(1).random(len(tr_all)) < 0.15
    parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
    h = {k: row_hash(pd.DataFrame(np.nan_to_num(A[i], nan=-9.0))) for k, i in parts.items()}
    keep, _ = dedup_across_splits(h, {k: y[i] for k, i in parts.items()}); parts = {k: i[keep[k]] for k, i in parts.items()}
    X, yy, Xv, yv, Xt = A[parts["train"]], y[parts["train"]], A[parts["validation"]], y[parts["validation"]], A[parts["test"]]
    it = stratified_sample(yy, min(len(yy), 2_000_000), np.random.default_rng(0), 1000); Xs_, ys_ = X[it], yy[it]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    b = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                          early_stopping_rounds=50, random_state=0, subsample=0.8, colsample_bytree=0.8)
    b.fit(Xs_, ys_, eval_set=[(Xv, yv)], verbose=False)
    Xn = np.nan_to_num(Xs_).astype(np.float32); mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xvn = ((np.nan_to_num(Xv) - mu) / sd).astype(np.float32)
    big = i23.train_mlp(((Xn - mu) / sd).astype(np.float32), np.asarray(ys_), Xvn, np.asarray(yv), (384, 384, 384, 384), 0, dev)
    small = i23.train_mlp(((Xn - mu) / sd).astype(np.float32), np.asarray(ys_), Xvn, np.asarray(yv), (64, 64), 0, dev, teacher=big)
    torch.save({"widths": (384, 384, 384, 384), "d": int(Xn.shape[1]), "state": big.cpu().state_dict()}, MD / "big.pt")
    torch.save({"widths": (64, 64), "d": int(Xn.shape[1]), "state": small.cpu().state_dict()}, MD / "small.pt")
    booster = b.get_booster(); booster.save_model(str(MD / "xgb.json"))
    np.savez(MD / "test_rows.npz", Xtt=((np.nan_to_num(Xt) - mu) / sd).astype(np.float32), Xraw=np.asarray(Xt, dtype=np.float32))
    json.dump({"device": str(dev), "train_seconds": time.time() - t0, "xgb_trees": int(b.best_iteration + 1),
               "n_train_sample": int(len(ys_)), "n_test": int(len(Xt))}, open(MD / "meta.json", "w"), indent=1)
    print("trained", json.load(open(MD / "meta.json")))


def time_it():
    import psutil
    import torch
    import xgboost as xgb
    from shield.eval.latency import _PinnedCore
    torch.set_num_threads(1)
    s_ = importlib.util.spec_from_file_location("i23", Path(__file__).with_name("41_iot23_shift.py"))
    i23 = importlib.util.module_from_spec(s_); s_.loader.exec_module(i23)
    def _mk(f):
        ck = torch.load(MD / f, weights_only=False); m = i23.MLP(ck["d"], ck["widths"]); m.load_state_dict(ck["state"]); return m.eval()
    small, big = _mk("small.pt"), _mk("big.pt")
    booster = xgb.Booster(); booster.load_model(str(MD / "xgb.json")); booster.set_param({"device": "cpu", "nthread": 1})
    Z = np.load(MD / "test_rows.npz"); Xtt, Xraw = Z["Xtt"], Z["Xraw"]; n = len(Xtt)
    sig = lambda z: 1 / (1 + np.exp(-z))
    # fixed 2026-11-08: probability = softmax[:, 1] (as scripts/41), float64 so ent2 never sees p == 1.0 in float32 (NaN
    # entropy made the quantile threshold NaN and suppressed every escalation in the first timing run)
    stu = lambda x: torch.softmax(small(torch.from_numpy(x)), 1).numpy().astype(np.float64)[:, 1]
    def expert(xs, xr):
        pm = torch.softmax(big(torch.from_numpy(xs)), 1).numpy()[:, 1] if big(torch.from_numpy(xs[:1])).shape[1] == 2 else sig(big(torch.from_numpy(xs)).numpy().ravel())
        return ((pm + booster.inplace_predict(xr)) / 2 > 0.5)
    out = {"core": 0, "cpu_percent_before": psutil.cpu_percent(1.0), "n_test": int(n)}
    rng = np.random.default_rng(0)
    with _PinnedCore(0), torch.inference_mode():
        H = np.concatenate([ent2(stu(Xtt[i:i + 4096])) for i in range(0, n, 4096)])
        assert np.isfinite(H).all(), "non-finite student entropy"
        sep = {}
        for b, reps in ((256, 300), (1, 3000)):
            idx = rng.integers(0, n - b, reps + 50); ts, tt = [], []
            for k, s in enumerate(idx):
                t0 = time.perf_counter(); stu(Xtt[s:s + b]); t1 = time.perf_counter(); expert(Xtt[s:s + b], Xraw[s:s + b]); t2 = time.perf_counter()
                if k >= 50:
                    ts.append(t1 - t0); tt.append(t2 - t1)
            sep[b] = {"mu_s": b / float(np.median(ts)), "mu_t": b / float(np.median(tt))}
        curve = {}
        for e in ES:
            thr = np.inf if e == 0 else float(np.quantile(H, 1 - e))
            idx = rng.integers(0, n - 256, 350); tb = []
            for k, s in enumerate(idx):
                t0 = time.perf_counter(); m = ent2(stu(Xtt[s:s + 256])) > thr
                if m.any():
                    expert(Xtt[s:s + 256][m], Xraw[s:s + 256][m])
                if k >= 50:
                    tb.append(time.perf_counter() - t0)
            idx = rng.integers(0, n, 3050); tl, esc = [], 0
            for k, s in enumerate(idx):
                t0 = time.perf_counter()
                if ent2(stu(Xtt[s:s + 1]))[0] > thr:
                    expert(Xtt[s:s + 1], Xraw[s:s + 1]); esc += k >= 50
                if k >= 50:
                    tl.append(time.perf_counter() - t0)
            tl = np.array(tl) * 1e3
            f = {b: 1 / (1 / sep[b]["mu_s"] + e / sep[b]["mu_t"]) for b in (256, 1)}
            curve[str(e)] = {"b256_throughput": 256 / float(np.median(tb)), "b256_formula": f[256],
                             "b1_throughput": 1e3 / float(tl.mean()), "b1_formula": f[1], "b1_p50_ms": float(np.percentile(tl, 50)),
                             "b1_p95_ms": float(np.percentile(tl, 95)), "b1_p99_ms": float(np.percentile(tl, 99)),
                             "b1_escalated_share": esc / 3000}
            c = curve[str(e)]
            print(f"iot23 e={e:.2f} b256 {c['b256_throughput']:.0f}/s (formula {c['b256_formula']:.0f}) b1 {c['b1_throughput']:.0f}/s "
                  f"(formula {c['b1_formula']:.0f}) p50 {c['b1_p50_ms']:.3f} p99 {c['b1_p99_ms']:.3f} ms esc {c['b1_escalated_share']:.3f}", flush=True)
    out.update({"separate": sep, "curve": curve, "cpu_percent_after": psutil.cpu_percent(1.0)})
    json.dump(out, open(OUT / "capacity_curve_iot23.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    train() if "--train" in sys.argv else time_it()
