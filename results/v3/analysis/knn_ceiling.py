"""Step 3b: nearest-neighbour overlap estimate (Cover & Hart 1967), validation queries vs train reference.

R_1NN -> Bayes error R* is bounded asymptotically by
    ((C-1)/C) * (1 - sqrt(1 - C/(C-1) * R_1NN))  <=  R*  <=  R_1NN
Per class, the 1-NN confusion shows which classes overlap in feature space (not a model weakness).
Test data is never read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(r"D:\shield_run\data\processed")
JOBS = [("ciciot", "class34"), ("ciciot", "family"), ("ciciomt", "attack"), ("ciciomt", "category")]
N_QUERY = 200_000
dev = torch.device("cuda")
torch.manual_seed(0)
rng = np.random.default_rng(0)


def nn1(R, Q, chunk=2048, rchunk=200_000):
    """Index of the nearest train row for each query (squared Euclidean)."""
    Rn = (R * R).sum(1)
    best_d = torch.full((len(Q),), float("inf"), device=dev)
    best_i = torch.zeros(len(Q), dtype=torch.long, device=dev)
    for q0 in range(0, len(Q), chunk):
        q = Q[q0:q0 + chunk]
        qn = (q * q).sum(1, keepdim=True)
        for r0 in range(0, len(R), rchunk):
            d = qn + Rn[r0:r0 + rchunk][None] - 2 * q @ R[r0:r0 + rchunk].T
            v, i = d.min(1)
            upd = v < best_d[q0:q0 + chunk]
            best_d[q0:q0 + chunk][upd] = v[upd]
            best_i[q0:q0 + chunk][upd] = i[upd] + r0
    return best_i.cpu().numpy()


out = {}
cache = {}
for ds, task in JOBS:
    if len(sys.argv) > 1 and ds not in sys.argv[1:]:
        continue
    meta = json.load(open(ROOT / ds / "meta.json"))
    classes = meta["tasks"][task]
    C = len(classes)
    ytr = np.load(ROOT / ds / f"y_{task}_train.npy").astype(np.int64)
    yva = np.load(ROOT / ds / f"y_{task}_validation.npy").astype(np.int64)
    if ds not in cache:   # the neighbour search does not depend on the task: do it once per dataset
        R = torch.as_tensor(np.load(ROOT / ds / "X_train.npy"), dtype=torch.float32, device=dev)
        Xva = np.load(ROOT / ds / "X_validation.npy")
        qi = np.sort(rng.choice(len(Xva), min(N_QUERY, len(Xva)), replace=False))
        Q = torch.as_tensor(Xva[qi], dtype=torch.float32, device=dev)
        cache = {ds: (qi, nn1(R, Q))}
        del R, Q
        torch.cuda.empty_cache()
    qi, nbr = cache[ds]
    y, p = yva[qi], ytr[nbr]
    r = float((p != y).mean())
    lo = (C - 1) / C * (1 - np.sqrt(max(0.0, 1 - C / (C - 1) * r)))
    pcf = f1_score(y, p, average=None, labels=range(C))
    mf1 = float(np.mean(pcf[np.bincount(y, minlength=C) > 0]))
    conf = np.zeros((C, C), np.int64)
    np.add.at(conf, (y, p), 1)
    off = [(conf[a, b], classes[a], classes[b]) for a in range(C) for b in range(C) if a != b]
    print(f"\n== {ds}/{task}: 1-NN error {100*r:.2f}%  ->  Bayes error in [{100*lo:.2f}%, {100*r:.2f}%]; "
          f"1-NN macro-F1 {100*mf1:.2f}")
    print("   lowest 1-NN F1:", ", ".join(f"{c} {100*f:.1f}" for f, c in sorted(zip(pcf, classes))[:8]))
    print("   top overlaps (true -> nearest):", ", ".join(f"{a}->{b} {n:,}" for n, a, b in sorted(off, reverse=True)[:6]))
    out[f"{ds}/{task}"] = {"n_query": int(len(y)), "err_1nn": r, "bayes_lower": float(lo), "macro_f1_1nn": mf1,
                           "per_class_f1_1nn": {c: float(f) for c, f in zip(classes, pcf)}}

dst = Path(__file__).with_name("knn_ceiling.json")
json.dump(out, open(dst, "w"), indent=1)
print("\nsaved", dst)
