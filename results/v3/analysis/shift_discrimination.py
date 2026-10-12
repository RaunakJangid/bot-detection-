"""Calibration vs discrimination under capture shift: does the FPR failure come from a shifted threshold (ranking
intact) or from lost ranking? Threshold recalibration cannot change ranking, so the paper must report both.
Benign-vs-attack ROC-AUC and PR-AUC (attack score = 1 - p(benign)) on validation (train captures) and test (new
captures), per model. CICIoMT2024 from cached logits; IoT-23 from saved scores (41_iot23_shift.py)."""
import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

P = Paths(False)
out = {}
for task in ("binary", "category", "attack"):
    d = load_processed(P.processed("ciciomt"), task, splits=("validation", "test"))
    ben = d.classes.index("Benign")
    yv, yt = (d.splits["validation"].y != ben).astype(int), (d.splits["test"].y != ben).astype(int)
    lc = P.v3 / "logits" / f"ciciomt_{task}"
    res = {}
    for f in sorted(glob.glob(str(lc / "*_validation.npy"))):
        m = os.path.basename(f)[:-len("_validation.npy")]
        if not (lc / f"{m}_test.npy").exists():
            continue
        sv = 1 - softmax(np.load(f).astype(np.float64))[:, ben]
        st = 1 - softmax(np.load(lc / f"{m}_test.npy").astype(np.float64))[:, ben]
        res[m] = {"val_roc": roc_auc_score(yv, sv), "test_roc": roc_auc_score(yt, st),
                  "val_pr": average_precision_score(yv, sv), "test_pr": average_precision_score(yt, st),
                  "test_prev": float(yt.mean())}
    out[f"ciciomt_{task}"] = res
for design in ("same_family", "capture_shift", "random_control"):
    f = P.v3 / "iot23" / f"scores_{design}.npz"
    if not f.exists():
        continue
    Z = np.load(f)
    out[f"iot23_{design}"] = {m: {"val_roc": roc_auc_score(Z["yv"], Z["sv"][i]), "test_roc": roc_auc_score(Z["yt"], Z["st"][i]),
                                  "val_pr": average_precision_score(Z["yv"], Z["sv"][i]),
                                  "test_pr": average_precision_score(Z["yt"], Z["st"][i]), "test_prev": float(Z["yt"].mean())}
                              for i, m in enumerate(Z["models"])}
json.dump(out, open(P.v3 / "shift" / "shift_discrimination.json", "w"), indent=2)
for k, res in out.items():
    tr = np.array([v["test_roc"] for v in res.values()]); vr = np.array([v["val_roc"] for v in res.values()])
    tp = np.array([v["test_pr"] for v in res.values()])
    best = max(res, key=lambda m: res[m]["test_roc"]); worst = min(res, key=lambda m: res[m]["test_roc"])
    print(f"{k:24s} n={len(res):2d} ROC val med {np.median(vr):.4f} -> test med {np.median(tr):.4f} "
          f"[min {tr.min():.4f} {worst}, max {tr.max():.4f} {best}] | PR test med {np.median(tp):.4f} (prev {list(res.values())[0]['test_prev']:.3f})")
