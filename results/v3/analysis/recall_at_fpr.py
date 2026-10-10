"""Operating point as deployed IDSs set it: the alarm threshold on p(attack) is chosen on VALIDATION benign flows
for a target false-positive rate; test recall / FPR / precision are then reported once. p(attack) = 1 - p(benign)
from each model's cached logits (binary task and the multi-class tasks)."""
from pathlib import Path

import numpy as np

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

P = Paths()
TARGETS = (0.001, 0.005, 0.01, 0.05)
rows = []
for ds, task in [("ciciot_full", "binary"), ("ciciot_full", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"),
                 ("botiot", "binary")]:
    lc = P.v3 / "logits" / f"{ds}_{task}"
    if not lc.exists():
        continue
    d = load_processed(P.processed(ds), task, splits=("validation", "test"))
    ben = next(i for i, c in enumerate(d.classes) if c in ("BenignTraffic", "Benign", "Normal", "0", 0))
    yv, yt = d.splits["validation"].y != ben, d.splits["test"].y != ben
    for m in ("shield_s0", "xgboost_s0", "ens4_s0", "seedexp_s0"):
        fv, ft = lc / f"{m}_validation.npy", lc / f"{m}_test.npy"
        if not ft.exists():
            continue
        st = 1 - softmax(np.load(ft).astype(np.float64))[:, ben]
        sv = 1 - softmax(np.load(fv).astype(np.float64))[:, ben] if fv.exists() else None
        for t in TARGETS:
            if sv is None:
                continue
            thr = np.quantile(sv[~yv], 1 - t)              # validation benign flows only
            alarm = st > thr
            rec, fpr = alarm[yt].mean(), alarm[~yt].mean()
            prec = (alarm & yt).sum() / max(1, alarm.sum())
            rows.append((ds, task, m, t, round(100 * rec, 2), round(100 * fpr, 3), round(100 * prec, 3)))
            print(*rows[-1], sep="\t", flush=True)
Path(__file__).with_suffix(".tsv").write_text("dataset\ttask\tmodel\ttarget_fpr\trecall\ttest_fpr\tprecision\n" +
                                               "\n".join("\t".join(map(str, r)) for r in rows))
