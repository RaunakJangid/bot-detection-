"""v3 step 8: detecting attack families never seen in training (leave-one-family-out).

For each attack family F of CICIoT2023 (family task) and CICIoMT2024 (category task) and each seed, a
binary student (all features, 128-64) is trained on attack-vs-benign WITHOUT any F rows (train and
validation). Detectors, each thresholded on benign VALIDATION flows at the same false-positive rate:
    student        the student's P(attack)
    +mahalanobis   union with a Mahalanobis novelty score on the student's hidden layer (benign fit)
    +flyhash       union with the fruit-fly FlyHash novelty filter on the input features (benign fit)
Reported on test: detection rate on F (unseen), on the other attacks (seen), and the benign FPR.
Writes outputs_v3/unseen/<dataset>_<F>_s<seed>.json and unseen/summary.csv.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from shield.cli import parser
from shield.data.datasets import Batcher, load_processed, predict_logits, stratified_sample
from shield.engine import fit
from shield.eval.metrics import softmax
from shield.eval.openset import FlyHash, Mahalanobis, hidden_features, threshold_at_fpr, union_score
from shield.models.student import StudentMLP
from shield.models.train import train_cfg_for
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything

log = get_logger("unseen")
JOBS = {"ciciot": "family", "ciciomt": "category"}


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--fpr", type=float, nargs="*", default=[0.01, 0.05])
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    paths, device = Paths(args.smoke), get_device(args.cpu)
    kcfg = load_config("kd", args.smoke)
    datasets = list(JOBS) if args.dataset in ("all", "main") else args.dataset.split(",")
    rows = []
    for ds in datasets:
        fam = load_processed(paths.processed(ds), JOBS[ds])
        tcfg = {**train_cfg_for(kcfg, ds), "epochs": args.epochs, "balance_power": 0.0, "samples_per_epoch": None}
        tr, va, te = fam.splits["train"], fam.splits["validation"], fam.splits["test"]
        benign = fam.classes.index("Benign")
        for F, fname in enumerate(fam.classes):
            if F == benign:
                continue
            for seed in args.seeds:
                out = paths.v3 / "unseen" / f"{ds}_{fname}_s{seed}.json"
                if out.exists() and not args.force:
                    rows.extend(load_json(out)["rows"])
                    continue
                seed_everything(seed)
                keep_tr, keep_va = np.flatnonzero(tr.y != F), np.flatnonzero(va.y != F)
                Xtr, ytr = np.asarray(tr.X[keep_tr]), (tr.y[keep_tr] != benign).astype(np.int64)
                Xva, yva = np.asarray(va.X[keep_va]), (va.y[keep_va] != benign).astype(np.int64)
                model = StudentMLP(fam.n_features, 2, [128, 64])
                ce = nn.CrossEntropyLoss()
                batcher = Batcher(Xtr, ytr, tcfg["batch_size"], device, seed=seed)
                fit(model, batcher, Xva, yva, lambda m, b: ce(m(b[0]), b[1]), tcfg, device, 2, f"{ds}-no{fname}-s{seed}")
                del batcher

                p_va = softmax(predict_logits(model, Xva, device))[:, 1]
                p_te = softmax(predict_logits(model, te.X, device))[:, 1]
                rng = np.random.default_rng(seed)
                ben_tr = np.flatnonzero(ytr == 0)
                fit_idx = np.sort(rng.choice(ben_tr, min(len(ben_tr), 500_000), replace=False))
                maha = Mahalanobis().fit(hidden_features(model, Xtr[fit_idx], device))
                fly = FlyHash(fam.n_features, seed=seed).fit(Xtr[fit_idx])
                nov_va = {"mahalanobis": maha.score(hidden_features(model, Xva, device)), "flyhash": fly.score(Xva)}
                nov_te = {"mahalanobis": maha.score(hidden_features(model, te.X, device)), "flyhash": fly.score(te.X)}

                bva = yva == 0
                is_F, is_ben = te.y == F, te.y == benign
                seen = ~is_F & ~is_ben
                scores_va = {"student": p_va}
                scores_te = {"student": p_te}
                for name in nov_va:
                    scores_va[f"student+{name}"] = union_score(p_va[bva], nov_va[name][bva], p_va, nov_va[name])
                    scores_te[f"student+{name}"] = union_score(p_va[bva], nov_va[name][bva], p_te, nov_te[name])
                recs = []
                for det in scores_va:
                    for fpr in args.fpr:
                        t = threshold_at_fpr(scores_va[det][bva], fpr)
                        flag = scores_te[det] > t
                        recs.append({"dataset": ds, "unseen": fname, "seed": seed, "detector": det, "fpr_target": fpr,
                                     "detect_unseen": float(flag[is_F].mean()), "detect_seen": float(flag[seen].mean()),
                                     "test_fpr": float(flag[is_ben].mean())})
                lab = np.r_[np.zeros(is_ben.sum()), np.ones(is_F.sum())]
                auc = {f"auroc_{n}": float(roc_auc_score(lab, np.r_[s[is_ben], s[is_F]]))
                       for n, s in {"student": p_te, **nov_te}.items()}
                for r in recs:
                    r.update(auc)
                save_json({"rows": recs, "flyhash_bytes": fly.size_bytes()}, out)
                best = {r["detector"]: r["detect_unseen"] for r in recs if r["fpr_target"] == args.fpr[0]}
                log.info("%s unseen %s s%d @FPR %.0f%%: %s | AUROC %s", ds, fname, seed, 100 * args.fpr[0],
                         {k: round(v, 3) for k, v in best.items()}, {k: round(v, 3) for k, v in auc.items()})
                rows.extend(recs)
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    (paths.v3 / "unseen").mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.v3 / "unseen" / "summary.csv", index=False)


if __name__ == "__main__":
    main()
