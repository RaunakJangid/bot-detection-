"""Step 3: identifiability ceiling from exact label collisions (no model involved).

For each split, rows with an identical feature vector but different labels cannot be separated
by any classifier. The "oracle" gives every identical vector its majority label within that split,
which is the best any deterministic classifier could do on those rows.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(r"D:\shield_run\data\processed")
JOBS = {
    "ciciot": (["binary", "family", "class34"], ["train", "validation", "test"]),
    "ciciomt": (["binary", "category", "attack"], ["train", "validation", "test"]),
    "botiot": (["binary", "category"], ["test"]),
}


def row_groups(X):
    X = np.ascontiguousarray(X)
    v = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    _, inv = np.unique(v, return_inverse=True)
    return inv.ravel()


def oracle(g, y, n_cls):
    key = g.astype(np.int64) * n_cls + y
    uk, cnt = np.unique(key, return_counts=True)
    ug, lab = uk // n_cls, uk % n_cls
    order = np.lexsort((-cnt, ug))          # per group, the most frequent label first
    first = np.ones(len(order), bool)
    first[1:] = ug[order][1:] != ug[order][:-1]
    maj = np.empty(g.max() + 1, np.int64)
    maj[ug[order][first]] = lab[order][first]
    n_lab = np.bincount(ug, minlength=g.max() + 1)   # distinct labels per group
    return maj[g], n_lab[g] > 1


out = {}
for ds, (tasks, splits) in JOBS.items():
    if len(sys.argv) > 1 and ds not in sys.argv[1:]:
        continue
    meta = json.load(open(ROOT / ds / "meta.json"))
    for s in splits:
        X = np.load(ROOT / ds / f"X_{s}.npy")
        g = row_groups(X)
        n, n_uniq = len(g), g.max() + 1
        del X
        print(f"\n== {ds} / {s}: {n:,} rows, {n_uniq:,} unique vectors ({100*(1-n_uniq/n):.2f}% duplicate rows)")
        for t in tasks:
            y = np.load(ROOT / ds / f"y_{t}_{s}.npy").astype(np.int64)
            classes = meta["tasks"][t]
            pred, conflict = oracle(g, y, len(classes))
            mf1 = f1_score(y, pred, average="macro")
            pcf = f1_score(y, pred, average=None, labels=range(len(classes)))
            pairs = Counter(zip(y[pred != y], pred[pred != y]))
            print(f"  {t:9s} rows in label-conflicting groups: {100*conflict.mean():6.3f}%   "
                  f"oracle macro-F1 ceiling: {100*mf1:6.2f}")
            worst = sorted(zip(pcf, classes))[:6]
            print("     lowest ceiling classes:", ", ".join(f"{c} {100*f:.1f}" for f, c in worst))
            print("     top collision pairs (true -> oracle):",
                  ", ".join(f"{classes[a]}->{classes[b]} {c:,}" for (a, b), c in pairs.most_common(5)))
            out[f"{ds}/{s}/{t}"] = {"rows": int(n), "unique": int(n_uniq), "conflict_frac": float(conflict.mean()),
                                    "oracle_macro_f1": float(mf1),
                                    "per_class": {c: float(f) for c, f in zip(classes, pcf)}}

dst = Path(__file__).with_name("ceiling_" + "_".join(sys.argv[1:] or ["all"]) + ".json")
json.dump(out, open(dst, "w"), indent=1)
print("\nsaved", dst)
