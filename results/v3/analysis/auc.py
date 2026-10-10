"""Threshold-free metrics (macro one-vs-rest ROC-AUC and PR-AUC) from the cached TEST logits, as recommended for
imbalanced IDS evaluation; complements macro-F1 at the validation-chosen operating point."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

P = Paths()
rows = []
for ds, task in [("ciciot_full", "class34"), ("ciciot_full", "family"), ("ciciot_full", "binary"), ("ciciot", "class34"),
                 ("ciciomt", "category"), ("ciciomt", "attack")]:
    lc = P.v3 / "logits" / f"{ds}_{task}"
    if not lc.exists():
        continue
    y = load_processed(P.processed(ds), task, splits=("test",)).splits["test"].y
    n = int(y.max()) + 1
    for m in ("shield_s0", "xgboost_s0", "teacher_s0", "ens4_s0", "seedexp_s0", "specialist_s0"):
        f = lc / f"{m}_test.npy"
        if not f.exists():
            continue
        p = softmax(np.load(f).astype(np.float64))
        if n == 2:
            roc, pr = roc_auc_score(y, p[:, 1]), average_precision_score(y, p[:, 1])
        else:
            Y = np.eye(n)[y]
            ok = Y.sum(0) > 0
            roc = roc_auc_score(Y[:, ok], p[:, ok], average="macro")
            pr = average_precision_score(Y[:, ok], p[:, ok], average="macro")
        rows.append((ds, task, m, round(100 * roc, 2), round(100 * pr, 2)))
        print(*rows[-1], sep="\t", flush=True)
Path(__file__).with_suffix(".tsv").write_text("dataset\ttask\tmodel\troc_auc\tpr_auc\n" +
                                               "\n".join("\t".join(map(str, r)) for r in rows))
