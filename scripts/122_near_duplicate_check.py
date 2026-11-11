"""Internal check (not for the papers, read-only): how close are test flows to training flows?

For CICIoT2023 (class34, pre-split Kaggle copy himadri07/ciciot2023) and CICIoMT2024 (category; official test captures):
20,000 test and 20,000 validation rows (rng 0) are compared with the FULL training set. Features exactly as the models
see them: the processed arrays are already signed-log transformed and standardised with the train statistics
(preprocess.py). Exact nearest-neighbour Euclidean distance by chunked matrix products.
Validation rows come from the same captures as training (CICIoMT2024: carved from train; CICIoT2023: same random
split), so they are the in-distribution reference for what "close" means.
Reported per split: share of exact duplicates (distance 0), share within 0.05 / 0.25 / 1.0, distance quantiles, and the
accuracy of large model seed 0 and small model seed 0 on near (<= 0.05) and far (> 1.0) test rows, to see whether
closeness to training inflates the scores.
Output: outputs_v5/near_duplicate_check.json."""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/near_duplicate_check.json")
N, TCH, RCH = 20_000, 500, 250_000
SETS = (("ciciot", "class34"), ("ciciomt", "category"))
CUTS = (1e-6, 0.05, 0.25, 1.0)          # 1e-6: exact duplicate up to rounding


def nn_dist(Q, Xtr):
    """float64 for the products: d2 = q2 + r2 - 2qr loses exact zeros in float32"""
    Q = np.asarray(Q, np.float64); best = np.full(len(Q), np.inf); q2 = (Q ** 2).sum(1)
    for j in range(0, len(Xtr), RCH):
        R = np.asarray(Xtr[j:j + RCH], np.float64); r2 = (R ** 2).sum(1)
        for i in range(0, len(Q), TCH):
            d2 = q2[i:i + TCH, None] + r2[None, :] - 2.0 * (Q[i:i + TCH] @ R.T)
            best[i:i + TCH] = np.minimum(best[i:i + TCH], d2.min(1))
    return np.sqrt(np.maximum(best, 0.0))


def main():
    from shield.data.datasets import load_processed
    from shield.utils.config import Paths
    P = Paths(False); res = {}
    for ds, task in SETS:
        t = time.time(); d = load_processed(P.processed(ds), task)
        Xtr = d.splits["train"].X; rng = np.random.default_rng(0); out = {}
        lc = P.v3 / "logits" / f"{ds}_{task}"
        for split in ("validation", "test"):
            S = d.splits[split]; idx = np.sort(rng.choice(len(S.y), N, replace=False))
            dist = nn_dist(S.X[idx], Xtr); y = S.y[idx]
            o = {"n": N, **{f"share_le_{c}": float((dist <= c).mean()) for c in CUTS},
                 "quantiles": {q: float(np.quantile(dist, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
            for name in ("teacher_s0", "shield_s0"):
                p = np.load(lc / f"{name}_{split}.npy", mmap_mode="r")[idx].argmax(1)
                near, far = dist <= 0.05, dist > 1.0
                o[f"acc_{name}"] = {"all": float((p == y).mean()),
                                    "near": float((p[near] == y[near]).mean()) if near.any() else None,
                                    "far": float((p[far] == y[far]).mean()) if far.any() else None,
                                    "n_near": int(near.sum()), "n_far": int(far.sum())}
            out[split] = o
            print(f"{ds}_{task} {split}: exact {100 * o['share_le_1e-06']:.1f}% <=0.05 {100 * o['share_le_0.05']:.1f}% "
                  f"<=1 {100 * o['share_le_1.0']:.1f}% median {o['quantiles'][0.5]:.3f} | teacher acc near/far "
                  f"{o['acc_teacher_s0']['near']} / {o['acc_teacher_s0']['far']} ({time.time() - t:.0f} s)", flush=True)
        res[f"{ds}_{task}"] = out
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
