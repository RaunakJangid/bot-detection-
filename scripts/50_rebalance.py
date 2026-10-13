"""Class rebalancing test: undersample frequent classes to a threshold T, SMOTE rare classes up to T (TRAIN split only).

Data: ciciot_fullho, 34 classes (train = full CICIoT2023 minus the 2M holdout; validation = strict validation rows;
test = the 2M holdout, scored ONCE for the validation-chosen arm and the baseline).
SMOTE (Chawla et al. 2002), implemented here: synthetic = x + u * (nn - x), x a real row of the class, nn one of its
5 nearest same-class neighbours (fitted on <= 50k real rows of that class), u ~ U(0, 1). Synthetic rows per class are
capped at 19x the real count (rare classes have ~1k rows and noisy labels).
Arms (XGBoost, same settings as the full-data baseline: depth 10, lr 0.1, <= 2000 trees, early stopping on a
validation subset):
  base       stratified cap 10M rows (the existing recipe)
  weights    same rows, sample weights count^-0.5
  under_T    classes above T randomly reduced to T, others unchanged
  smote_T    under_T + SMOTE up to T
  T in {20k, 50k, 100k, 200k}
Selection: highest validation macro-F1. Reported: validation macro-F1, benign false-alarm rate (benign predicted as any
attack), rare-class F1; holdout macro-F1 for base and the chosen arm."""

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("rebalance")
TS = (20_000, 50_000, 100_000, 200_000)
RARE = ["Backdoor_Malware", "BrowserHijacking", "CommandInjection", "DictionaryBruteForce", "Recon-PingSweep",
        "SqlInjection", "Uploading_Attack", "XSS"]


def smote(X, n_new, rng, k=5):
    base = X if len(X) <= 50_000 else X[rng.choice(len(X), 50_000, replace=False)]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(base))).fit(base)
    i = rng.integers(0, len(base), n_new)
    nb = nn.kneighbors(base[i], return_distance=False)[:, 1:]
    j = nb[np.arange(n_new), rng.integers(0, nb.shape[1], n_new)]
    u = rng.random((n_new, 1)).astype(np.float32)
    return (base[i] + u * (base[j] - base[i])).astype(np.float32)


def fit(X, y, w, Xv, yv, n):
    clf = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=50, random_state=0, objective="multi:softprob", num_class=n)
    clf.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    return clf


def predict(clf, X):
    return np.vstack([clf.predict_proba(X[i:i + 500_000]) for i in range(0, len(X), 500_000)]).argmax(1)


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    d = load_processed(P.processed("ciciot_fullho"), "class34")
    tr, va, te = d.splits["train"], d.splits["validation"], d.splits["test"]
    cls, n = d.classes, d.n_classes
    ben = cls.index("BenignTraffic"); rare = [cls.index(c) for c in RARE]
    counts = np.bincount(tr.y, minlength=n)
    log.info("train class counts: %s", {cls[c]: int(counts[c]) for c in np.argsort(counts)})
    rng = np.random.default_rng(0)
    iv = stratified_sample(va.y, 500_000, rng, 200)
    Xv_es, yv_es = np.asarray(va.X[iv], np.float32), va.y[iv]
    Xv, yv = np.asarray(va.X, np.float32), va.y

    def score(pred, y):
        b = y == ben
        return {"macro_f1": float(100 * f1_score(y, pred, labels=np.arange(n), average="macro", zero_division=0)),
                "benign_false_alarm": float(100 * (pred[b] != ben).mean()),
                "rare_f1": {cls[c]: float(100 * f1_score(y == c, pred == c, zero_division=0)) for c in rare}}

    res = {"counts": {cls[c]: int(counts[c]) for c in range(n)}, "arms": {}}
    arms = {}
    ib = stratified_sample(tr.y, 10_000_000, rng, 1000)
    arms["base"] = (ib, None)
    w = counts.astype(float) ** -0.5; w = w / (w[tr.y[ib]].mean())
    arms["weights"] = (ib, w)
    for T in TS:
        sel = np.concatenate([np.flatnonzero(tr.y == c) if counts[c] <= T else rng.choice(np.flatnonzero(tr.y == c), T, replace=False)
                              for c in range(n) if counts[c] > 0])
        arms[f"under_{T}"] = (np.sort(sel), None)
        arms[f"smote_{T}"] = (np.sort(sel), "smote", T)
    models = {}
    for name, spec in arms.items():
        idx = spec[0]
        X, y = np.asarray(tr.X[idx], np.float32), tr.y[idx]
        sw = spec[1][y] if isinstance(spec[1], np.ndarray) else None
        n_syn = 0
        if isinstance(spec[1], str) and spec[1] == "smote":
            T = spec[2]; parts_X, parts_y = [X], [y]
            for c in range(n):
                m = counts[c]
                if 0 < m < T:
                    k_new = int(min(T - m, 19 * m))
                    parts_X.append(smote(X[y == c], k_new, rng)); parts_y.append(np.full(k_new, c)); n_syn += k_new
            X, y = np.concatenate(parts_X), np.concatenate(parts_y)
        clf = fit(X, y, sw, Xv_es, yv_es, n)
        r = score(predict(clf, Xv), yv); r.update({"train_rows": int(len(y)), "synthetic_rows": int(n_syn)})
        res["arms"][name] = r; models[name] = clf
        log.info("%s: rows %d (synthetic %d) | val macro-F1 %.2f | benign false alarms %.2f%%", name, len(y), n_syn,
                 r["macro_f1"], r["benign_false_alarm"])
        save_json(res, P.v3 / "rebalance" / "rebalance.json")
    best = max(res["arms"], key=lambda k: res["arms"][k]["macro_f1"])
    Xt = np.asarray(te.X, np.float32)
    res["chosen"] = best
    res["holdout"] = {k: score(predict(models[k], Xt), te.y) for k in dict.fromkeys(["base", best])}
    log.info("chosen on validation: %s | holdout: %s", best,
             {k: (round(v["macro_f1"], 2), round(v["benign_false_alarm"], 2)) for k, v in res["holdout"].items()})
    save_json(res, P.v3 / "rebalance" / "rebalance.json")


if __name__ == "__main__":
    main()
