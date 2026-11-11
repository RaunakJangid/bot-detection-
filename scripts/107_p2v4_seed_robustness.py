"""P2 v4: robustness of the expert-versus-student comparison across training seeds of BOTH models (evidence check).

Earlier analyses (scripts/76, 85, 99) used one expert seed (teacher_s0 / ensemble_s0). Here:
  CIC (CICIoT2023 34-class, CICIoMT2024 binary / 6-class / 19-class): students shield_s0-4 x teachers teacher_s0-4
      (25 pairs) on the official test split;
  IoT-23 (capture_shift 1-1, 35-1; same_family 49-1, 7-1): students small_mlp_kd_s0-4 x experts ensemble_s0-2, where
      ensemble_sj = 0.5 (xgboost_sj + large_mlp_sj), the definition of ensemble_s0 in scripts/41 (15 pairs).
Per pair: expert - student macro-F1 and accuracy (sampled test distribution, as all earlier P2 results).
Per setting: all pair differences, matched-seed differences, counts favouring each model (|diff| > 0.05 points = tie
band), median and range. Uncertainty for the matched pair (s0, e0):
  CIC: paired bootstrap over flows (1,000 resamples; no flow times exist);
  IoT-23: paired moving-block bootstrap in time order (1,000 resamples) with block lengths 50, 200 and 1,000 flows
  (sensitivity to temporal dependence).
Output: outputs_v5/p2/seed_robustness.json."""
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
rng = np.random.default_rng(107)


def macro_f1(y, p, K):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    present = (cm.sum(1) + cm.sum(0)) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    return 100 * f1[present].mean()


def diff_stats(y, ps, pe, K):
    return macro_f1(y, pe, K) - macro_f1(y, ps, K), 100 * ((pe == y).mean() - (ps == y).mean())


def summarise(pairs, matched):
    d = [x[0] for x in pairs]; a = [x[1] for x in pairs]
    sign = lambda v: ("expert" if v > 0.05 else "student" if v < -0.05 else "tie")
    return {"n_pairs": len(pairs), "f1_diff_all": [round(x, 3) for x in d], "f1_diff_matched": [round(x[0], 3) for x in matched],
            "acc_diff_matched": [round(x[1], 3) for x in matched],
            "favour_counts_all": {k: sum(sign(x) == k for x in d) for k in ("expert", "student", "tie")},
            "f1_diff_median": float(np.median(d)), "f1_diff_range": [float(min(d)), float(max(d))],
            "acc_diff_median": float(np.median(a)), "acc_diff_range": [float(min(a)), float(max(a))]}


def boot_iid(y, ps, pe, K, B=1000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); v.append(diff_stats(y[i], ps[i], pe[i], K)[0])
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def boot_block(y, ps, pe, K, L, B=1000):
    n = len(y); L = min(L, n); nb = int(np.ceil(n / L)); v = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, nb); i = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        v.append(diff_stats(y[i], ps[i], pe[i], K)[0])
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def main():
    from shield.data.datasets import load_processed
    from shield.utils.config import Paths
    P = Paths(False); res = {}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        yt = load_processed(P.processed(ds), task, splits=("test",)).splits["test"].y.astype(np.int64)
        lc = P.v3 / "logits" / f"{ds}_{task}"; K = int(yt.max()) + 1
        PS = [np.load(lc / f"shield_s{s}_test.npy").argmax(1) for s in range(5)]
        PE = [np.load(lc / f"teacher_s{t}_test.npy").argmax(1) for t in range(5)]
        K = max(K, max(int(p.max()) for p in PS + PE) + 1)
        pairs = [diff_stats(yt, PS[s], PE[t], K) for s in range(5) for t in range(5)]
        matched = [diff_stats(yt, PS[s], PE[s], K) for s in range(5)]
        r = summarise(pairs, matched); r["ci95_matched_s0_t0_iid"] = boot_iid(yt, PS[0], PE[0], K)
        res[f"{ds}_{task}"] = r
        print(ds, task, json.dumps({k: r[k] for k in ("f1_diff_matched", "favour_counts_all", "f1_diff_median", "f1_diff_range",
                                                     "acc_diff_median", "ci95_matched_s0_t0_iid")}), flush=True)
    for design, cap in (("capture_shift", "CTU-IoT-Malware-Capture-1-1"), ("capture_shift", "CTU-IoT-Malware-Capture-35-1"),
                        ("same_family", "CTU-IoT-Malware-Capture-49-1"), ("same_family", "CTU-IoT-Malware-Capture-7-1")):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        m = Z["capture"] == cap; o = np.argsort(Z["ts"][m], kind="stable"); yt = Z["yt"][m][o].astype(np.int64)
        st = lambda name: Z["st"][models.index(name)][m][o]
        PS = [(st(f"small_mlp_kd_s{s}") > 0.5).astype(np.int64) for s in range(5)]
        PE = [((0.5 * (st(f"xgboost_s{j}") + st(f"large_mlp_s{j}"))) > 0.5).astype(np.int64) for j in range(3)]
        chk = bool(np.array_equal(PE[0], (st("ensemble_s0") > 0.5).astype(np.int64)))
        pairs = [diff_stats(yt, PS[s], PE[j], 2) for s in range(5) for j in range(3)]
        matched = [diff_stats(yt, PS[s], PE[s % 3], 2) for s in range(5)]
        r = summarise(pairs, matched); r["ensemble_s0_reconstruction_matches"] = chk
        r["ci95_matched_s0_e0_block"] = {str(L): boot_block(yt, PS[0], PE[0], 2, L) for L in (50, 200, 1000)}
        r["ci95_matched_s0_e0_iid"] = boot_iid(yt, PS[0], PE[0], 2); r["n_flows"] = int(len(yt))
        res[f"{design}|{cap}"] = r
        print(cap, json.dumps({k: r[k] for k in ("f1_diff_matched", "favour_counts_all", "f1_diff_median", "f1_diff_range",
                                                "acc_diff_median", "ensemble_s0_reconstruction_matches", "ci95_matched_s0_e0_iid",
                                                "ci95_matched_s0_e0_block")}), flush=True)
    json.dump(res, open(OUT / "seed_robustness.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
