"""Check behind row 28ag: did the tiny student actually see IAT, and why does it not exploit the leak?

Cells (IAT kept, de-dup on): CICIoT2023 34-class (official = random split) and CICIoMT2024 attack (random split).
Teacher as in scripts/55_student_protocol.py. Reported: IAT's rank in the teacher's gain importance and whether it is
in the student's top-k. Student variants (2 seeds, same recipe otherwise):
  topk_slog     the original recipe (top-k by gain, signed-log + z-score)
  iat_slog      top-k with IAT forced in, signed-log + z-score
  iat_zraw      top-k with IAT forced in, plain z-score (no log)
  iat_rank      top-k with IAT forced in, IAT replaced by its rank among training values (quantile transform)
A student that gains with a scaling that keeps IAT's fine structure means the original recipe is protected by its
input scaling, not by the model size."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import f1_score

from shield.cli import parser
from shield.data.datasets import stratified_sample
from shield.data.preprocess import LABEL, cic_task_specs, dedup_across_splits, row_hash
from shield.utils.config import Paths, load_config
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("iat_check")
_s = importlib.util.spec_from_file_location("sp", Path(__file__).with_name("55_student_protocol.py"))
sp = importlib.util.module_from_spec(_s); _s.loader.exec_module(sp)
CELLS = [("ciciot", "class34", 32, "official"), ("ciciomt", "attack", 32, "random")]


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P, cfg = Paths(False), load_config("data", False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for src, task, k, split in CELLS:
        fr = sp.e4.frames_for(src, P, cfg, split)
        labels = sorted(set().union(*(set(df[LABEL].unique()) for df in fr.values())))
        classes, fn = cic_task_specs(src, labels, False)[task]
        feats = [c for c in fr["train"].columns if c != LABEL]
        lab_index = {l: i for i, l in enumerate(labels)}
        codes = {s: df[LABEL].map(lab_index).to_numpy(np.int64) for s, df in fr.items()}
        keep = dedup_across_splits({s: row_hash(df[feats]) for s, df in fr.items()}, codes)[0]
        X = {s: fr[s][feats].to_numpy(np.float32)[keep[s]] for s in fr}
        for s in X:
            X[s][~np.isfinite(X[s])] = 0
        ci = {c: i for i, c in enumerate(classes)}
        lut = np.array([ci[fn(l)] for l in labels])
        y = {s: lut[codes[s][keep[s]]] for s in fr}
        n = len(classes)
        rng = np.random.default_rng(0)
        it = stratified_sample(y["train"], min(2_000_000, len(y["train"])), rng, 1000)
        iv = stratified_sample(y["validation"], min(500_000, len(y["validation"])), rng, 200)
        tch = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                                random_state=0, early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.8,
                                objective="multi:softprob", num_class=n)
        tch.fit(X["train"][it], y["train"][it], eval_set=[(X["validation"][iv], y["validation"][iv])], verbose=False)
        imp = tch.feature_importances_
        order = np.argsort(-imp)
        iat = feats.index("IAT")
        rank = int(np.flatnonzero(order == iat)[0]) + 1
        cell = f"{src}_{task}|{split}"
        res = {"iat_gain_rank": rank, "iat_in_topk": bool(rank <= k), "iat_gain_share": float(imp[iat] / imp.sum())}
        log.info("%s: IAT gain rank %d of %d (share %.3f), in top-%d: %s", cell, rank, len(feats), res["iat_gain_share"], k, rank <= k)
        proba = lambda A: np.vstack([tch.predict_proba(A[i:i + 500_000]) for i in range(0, len(A), 500_000)])
        Ptr = proba(X["train"][it]).astype(np.float32)
        top = list(order[:k])
        top_iat = top if iat in top else top[:-1] + [iat]
        ranks = np.sort(X["train"][it][:, iat])
        for variant, cols, mode in (("topk_slog", top, "slog"), ("iat_slog", top_iat, "slog"),
                                    ("iat_zraw", top_iat, "raw"), ("iat_rank", top_iat, "rank")):
            def tf(A):
                Z = A[:, cols].astype(np.float64)
                if mode == "slog":
                    Z = sp.slog(Z)
                elif mode == "rank":
                    Z = sp.slog(Z)
                    j = cols.index(iat)
                    Z[:, j] = np.searchsorted(ranks, A[:, iat]) / len(ranks)
                return Z
            Ztr = tf(X["train"][it]); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
            z = lambda A: ((tf(A) - mu) / sd).astype(np.float32)
            Zv, Zt = z(X["validation"][iv]), z(X["test"])
            f1s = []
            for s in (0, 1):
                m = sp.train_student(((Ztr - mu) / sd).astype(np.float32), y["train"][it], Ptr, Zv, y["validation"][iv], n, s, dev)
                with torch.no_grad():
                    T = torch.tensor(Zt, device=dev)
                    ps = torch.cat([m(T[i:i + 200_000]).argmax(1) for i in range(0, len(T), 200_000)]).cpu().numpy()
                f1s.append(float(100 * f1_score(y["test"], ps, average="macro")))
            res[variant] = f1s
            log.info("%s %s: student macro-F1 %s", cell, variant, [round(f, 2) for f in f1s])
            out[cell] = res
            save_json(out, P.v3 / "ablation" / "student_iat_check.json")


if __name__ == "__main__":
    main()
