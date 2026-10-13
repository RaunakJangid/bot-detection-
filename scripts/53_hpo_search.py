"""Step 4: can a metaheuristic hyper-parameter search (GWO + SA, the published hybrid that won the placement benchmark)
find a detector that beats the current best, compared with random search and TPE at the SAME evaluation budget?

Data: ciciot_fullho, 34 classes (train = full CICIoT2023 minus the 2M holdout; validation = strict validation rows;
test = the 2M holdout, scored ONCE per finalist).
Model: XGBoost (GPU hist), no size limit. Search space: 8 levels per dimension (grid index x in {0..7}^8):
  max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_lambda, max_bin, class-weight
  exponent alpha (sample weight count^-alpha). x0 = the existing recipe (depth 10, lr 0.1, alpha 0, others default).
Search: every method gets 25 evaluations, the same 1M-row stratified training sample (>= 1000 per class), early
stopping on a fixed per-class 30% of validation, objective = macro-F1 on the disjoint 70% (validation has 728k rows).
  random   x0 then uniform grid points
  tpe      independent TPE (Bergstra et al. 2011; Optuna's default form), implemented here: 8 startup points
           (x0 + random), then per dimension Parzen estimators over the grid for the top 25% (l) and the rest (g)
           with a +1 prior, 24 candidates drawn from l, the one maximising l/g evaluated
  gwo_sa   shield.eval.rule_search.gwo_sa, population 5, last 30% of the budget SA polish
Final: each method's best point is retrained on the 10M-row stratified training sample (the base recipe's rows),
scored on the 70% validation part, then once on the holdout. The validation-best finalist is refitted with seeds 1 and 2.
Win rule (pre-set): holdout macro-F1 >= current best (experts + ens4, 75.4) + 0.5. Also reported vs the base XGB
(holdout 71.34, row 28z)."""

import json
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.data.preprocess import stratified_split
from shield.eval.rule_search import Budget, Objective, gwo_sa, random_search
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("hpo")
GRID = {
    "max_depth": [6, 8, 10, 12, 14, 16, 18, 20],
    "learning_rate": [0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.25, 0.3],
    "min_child_weight": [0.5, 1, 2, 4, 8, 16, 32, 64],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    "max_bin": [64, 128, 192, 256, 384, 512, 768, 1024],
    "alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
}
KEYS = list(GRID)
D, G, BUDGET = len(KEYS), 8, 25
X0 = np.array([2, 3, 1, 7, 7, 4, 3, 0])


def decode(x):
    return {k: GRID[k][int(v)] for k, v in zip(KEYS, x)}


def tpe(obj, D, G, x0, rng, n_start=8, gamma=0.25, n_cand=24):
    H = [(x0.copy(), obj(x0))]
    for _ in range(n_start - 1):
        x = rng.integers(0, G, D); H.append((x, obj(x)))
    while True:
        Xh = np.array([h[0] for h in H]); F = np.array([h[1] for h in H])
        n_good = max(1, int(np.ceil(gamma * len(H))))
        good = np.argsort(-F)[:n_good]; bad = np.setdiff1d(np.arange(len(H)), good)
        l = np.ones((D, G)); g = np.ones((D, G))
        for d in range(D):
            np.add.at(l[d], Xh[good, d], 1); np.add.at(g[d], Xh[bad, d], 1)
        l /= l.sum(1, keepdims=True); g /= g.sum(1, keepdims=True)
        C = np.array([[rng.choice(G, p=l[d]) for d in range(D)] for _ in range(n_cand)])
        score = np.log(l[np.arange(D), C]).sum(1) - np.log(g[np.arange(D), C]).sum(1)
        x = C[score.argmax()]; H.append((x, obj(x)))


def fit(p, X, y, counts, Xe, ye, n, seed=0):
    w = counts.astype(float) ** -p["alpha"]; w = w[y] / w[y].mean()
    c = xgb.XGBClassifier(n_estimators=3000, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                          min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                          colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], max_bin=p["max_bin"],
                          tree_method="hist", device="cuda", early_stopping_rounds=50, random_state=seed,
                          objective="multi:softprob", num_class=n)
    c.fit(X, y, sample_weight=w, eval_set=[(Xe, ye)], verbose=False)
    return c


def predict(c, X):
    return np.vstack([c.predict_proba(X[i:i + 500_000]) for i in range(0, len(X), 500_000)]).argmax(1)


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    out_f = P.v3 / "hpo" / "hpo_search.json"
    d = load_processed(P.processed("ciciot_fullho"), "class34")
    tr, va, te = d.splits["train"], d.splits["validation"], d.splits["test"]
    n, cls = d.n_classes, d.classes
    ben = cls.index("BenignTraffic")
    counts = np.bincount(tr.y, minlength=n)
    rng = np.random.default_rng(0)
    parts = stratified_split(va.y, (0.3, 0.0, 0.7), rng)                  # per-class 30/70 split of validation
    i_es, i_obj = np.sort(parts["train"]), np.sort(parts["test"])
    Xe, ye = np.asarray(va.X[i_es], np.float32), va.y[i_es]
    Xo, yo = np.asarray(va.X[i_obj], np.float32), va.y[i_obj]
    i_s = stratified_sample(tr.y, 1_000_000, rng, 1000)
    Xs, ys = np.asarray(tr.X[i_s], np.float32), tr.y[i_s]
    try:
        res = json.loads(out_f.read_text())
    except (FileNotFoundError, ValueError):
        res = {"grid": GRID, "budget": BUDGET, "search": {}, "final": {}}
    cache = {}
    for m in res["search"].values():
        for e in m["evals"]:
            cache[tuple(e["x"])] = e["val_f1"]

    def f(x):
        k = tuple(int(v) for v in x)
        if k in cache:
            return cache[k]
        t = time.time()
        c = fit(decode(x), Xs, ys, counts, Xe, ye, n)
        v = f1_score(yo, predict(c, Xo), average="macro")
        cache[k] = v
        log.info("  %s -> %.4f (%d trees, %.0fs)", decode(x), v, c.best_iteration + 1, time.time() - t)
        return v

    for name, alg in (("random", random_search), ("tpe", tpe), ("gwo_sa", lambda o, D_, G_, x0, r: gwo_sa(o, D_, G_, x0, r, n=5))):
        if name in res["search"]:
            continue
        evals = []

        def g(x, evals=evals):
            v = f(x); evals.append({"x": [int(t) for t in x], "val_f1": float(v)}); return v

        obj = Objective(g, BUDGET)
        try:
            alg(obj, D, G, X0.copy(), np.random.default_rng(1))
        except Budget:
            pass
        res["search"][name] = {"best_x": [int(t) for t in obj.best_x], "best_params": decode(obj.best_x),
                               "best_val_f1": float(obj.best_f), "trace": obj.trace, "evals": evals}
        log.info("%s: best search val macro-F1 %.4f with %s", name, obj.best_f, decode(obj.best_x))
        save_json(res, out_f)
    # finalists on the 10M base rows
    del Xs, ys
    i_b = stratified_sample(tr.y, 10_000_000, np.random.default_rng(0), 1000)
    Xb, yb = np.asarray(tr.X[i_b], np.float32), tr.y[i_b]
    Xv, yv = Xo, yo                                                       # the 70% part (not used for early stopping)
    Xt = np.asarray(te.X, np.float32)

    def score(pred, y):
        return {"macro_f1": float(100 * f1_score(y, pred, labels=np.arange(n), average="macro", zero_division=0)),
                "benign_false_alarm": float(100 * (pred[y == ben] != ben).mean())}

    for name in ("random", "tpe", "gwo_sa"):
        key = f"{name}_s0"
        if key in res["final"]:
            continue
        p = res["search"][name]["best_params"]
        c = fit(p, Xb, yb, counts, Xe, ye, n)
        res["final"][key] = {"params": p, "trees": int(c.best_iteration + 1), "validation": score(predict(c, Xv), yv),
                             "holdout": score(predict(c, Xt), te.y)}
        log.info("final %s: val %s | holdout %s", key, res["final"][key]["validation"], res["final"][key]["holdout"])
        save_json(res, out_f)
    best = max(("random", "tpe", "gwo_sa"), key=lambda k: res["final"][f"{k}_s0"]["validation"]["macro_f1"])
    res["chosen"] = best
    for s in (1, 2):
        key = f"{best}_s{s}"
        if key in res["final"]:
            continue
        p = res["search"][best]["best_params"]
        c = fit(p, Xb, yb, counts, Xe, ye, n, seed=s)
        res["final"][key] = {"params": p, "trees": int(c.best_iteration + 1), "validation": score(predict(c, Xv), yv),
                             "holdout": score(predict(c, Xt), te.y)}
        log.info("final %s: val %s | holdout %s", key, res["final"][key]["validation"], res["final"][key]["holdout"])
        save_json(res, out_f)
    h = [res["final"][f"{best}_s{s}"]["holdout"]["macro_f1"] for s in range(3)]
    res["verdict"] = {"chosen": best, "holdout_mean": float(np.mean(h)), "holdout_sd": float(np.std(h)),
                      "win": bool(np.mean(h) >= 75.4 + 0.5)}
    log.info("verdict: %s", res["verdict"])
    save_json(res, out_f)


if __name__ == "__main__":
    main()
