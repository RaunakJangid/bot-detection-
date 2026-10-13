"""Comparability with Jin & Lee 2026 (5% FPR operating point): realised test FPR when the threshold is set at 5% FPR
on validation benign flows. CICIoMT (capture shift), CICIoT (control) from cached logits; IoT-23 designs from saved
scores (within-capture pooled). Median over models, and min/max."""
import glob
import os

import numpy as np

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

P = Paths(False)
for target in (0.05, 0.01):
    print(f"\n== threshold at {int(target*100)}% validation FPR")
    for ds, task in (("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack"), ("ciciot", "class34")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        ben = next(i for i, c in enumerate(d.classes) if c in ("Benign", "BenignTraffic"))
        bv, bt = d.splits["validation"].y == ben, d.splits["test"].y == ben
        lc = P.v3 / "logits" / f"{ds}_{task}"
        f = []
        for v in glob.glob(str(lc / "*_validation.npy")):
            t = v.replace("_validation", "_test")
            if not os.path.exists(t):
                continue
            sv = 1 - softmax(np.load(v).astype(np.float64)[bv])[:, ben]
            st = 1 - softmax(np.load(t).astype(np.float64)[bt])[:, ben]
            f.append(float((st > np.quantile(sv, 1 - target)).mean()))
        f = np.array(f)
        print(f"  {ds}_{task:9s} n={len(f):2d} test FPR median {100*np.median(f):5.2f}% [{100*f.min():5.2f}, {100*f.max():5.2f}]")
    for design in ("same_family", "capture_shift", "random_control"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz")
        f = np.array([float((Z["st"][i][Z["yt"] == 0] > np.quantile(Z["sv"][i][Z["yv"] == 0], 1 - target)).mean())
                      for i in range(len(Z["models"]))])
        print(f"  iot23_{design:14s} n={len(f):2d} test FPR median {100*np.median(f):5.2f}% [{100*f.min():5.2f}, {100*f.max():5.2f}]")
