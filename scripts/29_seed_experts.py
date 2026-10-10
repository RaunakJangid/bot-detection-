"""Seed replicates of the best detector (ens4 + rule + low-volume experts), no leakage.

Per seed s: LightGBM and CatBoost are retrained with seed s (row sampling and bagging differ); the neural
members (teacher, quantile ResMLPs) are the same seed-averaged members as ens4_s0 (disclosed); the Caruana
ensemble, the decision rule and the experts are rebuilt. Expert configurations, all chosen on VALIDATION:
  hand      one expert on the fixed low-volume group of 27_specialist.py
  single    one expert on benign + every class with validation F1 below --weak
  clusters  one expert per confusion cluster: connected components of the weak classes under the validation
            confusion graph (edge if the symmetric confusion rate > --edge); benign joins the cluster it is
            most confused with
each with or without label cleaning (TRAIN only: 2-fold cross-fitted expert, rows whose out-of-fold probability
of their own label is below --clean-tau are dropped). Test scored once for the chosen configuration; all
configurations' test scores are saved for the record, never used for choice."""

import importlib.util
import json

import numpy as np
import xgboost as xgb

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.eval.predictions import model_logits
from shield.utils.config import Paths
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("seed_experts")
SUB = 1.0   # row/column subsampling of the expert XGBoost (set by --subsample; < 1 makes seeds differ)
_spec = importlib.util.spec_from_file_location("dv4", __file__.replace("29_seed_experts", "19_detector_v4"))
dv4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dv4)
_spec = importlib.util.spec_from_file_location("sp", __file__.replace("29_seed_experts", "27_specialist"))
sp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sp)
SPLITS = ("validation", "test")


def clusters(yv, pv, n, weak, ben, edge):
    C = np.zeros((n, n)); np.add.at(C, (yv, pv), 1)
    R = C / np.maximum(C.sum(1, keepdims=True), 1)
    S = R + R.T
    comp, seen = [], set()
    for c in weak:
        if c in seen:
            continue
        stack, cur = [c], []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i); cur.append(i)
            stack += [j for j in weak if j not in seen and S[i, j] > edge]
        comp.append(sorted(cur))
    j = int(np.argmax([S[ben, g].sum() for g in comp]))
    comp[j] = sorted(comp[j] + [ben])
    return [g for g in comp if len(g) >= 2]


def fit_expert(X, y, Xv, yv, seed, power=0.5):
    k = int(y.max()) + 1
    cnt = np.bincount(y, minlength=k).astype(float)
    w = np.maximum(cnt, 1) ** -power; w = w / (w * cnt).sum() * cnt.sum()   # power 1 = fully balanced
    clf = xgb.XGBClassifier(n_estimators=3000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=100, random_state=seed, subsample=SUB, colsample_bytree=SUB,
                            objective="multi:softprob" if k > 2 else "binary:logistic")
    clf.fit(X, y, sample_weight=w[y], eval_set=[(Xv, yv)], verbose=False)
    return clf


def train_group(data, G, seed, clean, tau, max_benign, ben, power=0.5):
    tr, va = data.splits["train"], data.splits["validation"]
    rng = np.random.default_rng(seed)
    gmap = np.full(data.n_classes, -1); gmap[G] = np.arange(len(G))
    idx = np.flatnonzero(np.isin(tr.y, G))
    b = idx[tr.y[idx] == ben]
    if len(b) > max_benign:
        idx = np.sort(np.concatenate([idx[tr.y[idx] != ben], rng.choice(b, max_benign, replace=False)]))
    vi = np.flatnonzero(np.isin(va.y, G))
    es = vi[stratified_sample(va.y[vi], min(len(vi), 300_000), rng, 30)]
    X, y, Xv, yv = np.asarray(tr.X[idx]), gmap[tr.y[idx]], np.asarray(va.X[es]), gmap[va.y[es]]
    dropped = 0
    if clean:
        fold = rng.integers(0, 2, len(y))
        keep = np.ones(len(y), bool)
        for f in (0, 1):
            m = fit_expert(X[fold != f], y[fold != f], Xv, yv, seed, power)
            pr = m.predict_proba(X[fold == f])
            keep[np.flatnonzero(fold == f)] = pr[np.arange(len(pr)), y[fold == f]] >= tau
        dropped = int((~keep).sum())
        X, y = X[keep], y[keep]
    return fit_expert(X, y, Xv, yv, seed, power), dropped


def combine(pm, experts, X):
    out = pm.copy()
    for G, clf, lam in experts:
        ps = clf.predict_proba(X)
        mass = pm[:, G].sum(1, keepdims=True)
        out[:, G] = mass * (lam * ps + (1 - lam) * pm[:, G] / np.maximum(mass, 1e-12))
    return out


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--weak", type=float, default=0.8)
    p.add_argument("--edge", type=float, default=0.02)
    p.add_argument("--clean-tau", type=float, default=0.05)
    p.add_argument("--max-benign", type=int, default=2_000_000)
    p.add_argument("--gbdt-cap", type=int, default=6_000_000)
    p.add_argument("--subsample", type=float, default=1.0)
    args = p.parse_args()
    global SUB
    SUB = args.subsample
    tag = "" if SUB == 1.0 else f"_sub{SUB}"
    paths, device = Paths(args.smoke), get_device(args.cpu)
    ds, task = args.dataset, args.task
    data = load_processed(paths.processed(ds), task)
    n, yv, yt = data.n_classes, data.splits["validation"].y, data.splits["test"].y
    prior = np.bincount(data.splits["train"].y, minlength=n) / len(data.splits["train"].y)
    ben = next(i for i, c in enumerate(data.classes) if c in ("BenignTraffic", "Benign", "Normal"))
    Xv, Xt = np.asarray(data.splits["validation"].X), np.asarray(data.splits["test"].X)
    f1 = lambda y, P: classification_metrics(y, np.log(np.clip(P, 1e-12, 1)), n)["macro_f1"]
    neural = {"teacher": [model_logits(paths, data, "teacher", s, SPLITS, device) for s in range(5)],
              "xgboost": [model_logits(paths, data, "xgboost", 0, SPLITS, device)],
              "qt_ce": [dv4.load(paths, ds, task, f"qt_ce_s{s}") for s in range(3)],
              "qt_bs": [dv4.load(paths, ds, task, f"qt_bs_s{s}") for s in range(3)]}
    if dv4.have(paths, ds, task, "tabm_s0"):   # 25_deep_baselines.py --models tabm (TabM, ICLR 2025)
        neural["tabm"] = [dv4.load(paths, ds, task, "tabm_s0")]
    for seed in args.seeds:
        out_f = paths.v3 / "seed_experts" / f"{ds}_{task}_s{seed}{tag}.json"
        if out_f.exists() and not args.force:
            continue
        for name, fn in ((f"lgbm_s{seed}", lambda: dv4.run_lgbm(data, args.gbdt_cap, seed)),
                         (f"catboost_s{seed}", lambda: dv4.run_catboost(data, args.gbdt_cap, seed))):
            if not dv4.have(paths, ds, task, name):
                dv4.save(paths, ds, task, name, fn())
        members = {**neural, "lgbm": [dv4.load(paths, ds, task, f"lgbm_s{seed}")],
                   "catboost": [dv4.load(paths, ds, task, f"catboost_s{seed}")]}
        P = {m: {s: np.mean([softmax(z[s]) for z in zs], 0) for s in SPLITS} for m, zs in members.items()}
        counts, _ = dv4.ensemble_selection({m: P[m]["validation"] for m in P}, yv, n)
        tot = sum(counts.values())
        le = {s: np.log(np.clip(sum(c * P[m][s] for m, c in counts.items()) / tot, 1e-12, 1)) for s in SPLITS}
        bias = fit_decision_rule(le["validation"], yv, prior, device)["bias"]
        pm = {s: softmax(apply_bias(le[s], bias).astype(np.float64)) for s in SPLITS}
        base = {"val": f1(yv, pm["validation"]), "test": f1(yt, pm["test"])}
        log.info("s%d ens4+rule %s members %s", seed, base, counts)
        pc = classification_metrics(yv, np.log(np.clip(pm["validation"], 1e-12, 1)), n, data.classes, full=True)["per_class"]
        weak = [i for i, c in enumerate(data.classes) if i != ben and pc[c]["f1"] < args.weak]
        cfgs = {"hand": [np.array([data.classes.index(c) for c in sp.GROUP if c in data.classes])],
                "single": [np.array(sorted([ben] + weak))],
                "clusters": [np.array(g) for g in clusters(yv, pm["validation"].argmax(1), n, weak, ben, args.edge)]}
        res = {}
        variants = [(c, cl, 0.5) for c in cfgs for cl in (False, True)] + [("clusters", False, 1.0), ("clusters", True, 1.0)]
        for cname, clean, power in variants:
                groups = cfgs[cname]
                key = f"{cname}{'+clean' if clean else ''}{'+balanced' if power == 1.0 else ''}"
                experts, dropped = [], []
                for G in groups:
                    clf, d = train_group(data, G, seed, clean, args.clean_tau, args.max_benign, ben, power)
                    dropped.append(d)
                    best = max(((f1(yv, combine(pm["validation"], [(G, clf, lam)], Xv)), lam)
                                for lam in (0.0, 0.25, 0.5, 0.75, 1.0)))
                    experts.append((G, clf, best[1]))
                res[key] = {"groups": [[data.classes[i] for i in G] for G in groups], "lambdas": [e[2] for e in experts],
                            "dropped": dropped, "val": f1(yv, combine(pm["validation"], experts, Xv)),
                            "test": f1(yt, combine(pm["test"], experts, Xt))}
                log.info("s%d %s val %.4f test %.4f groups %s", seed, key, res[key]["val"], res[key]["test"],
                         res[key]["groups"])
                res[key]["_experts"] = experts
        chosen = max(res, key=lambda k: res[k]["val"])
        zt = np.log(np.clip(combine(pm["test"], res[chosen]["_experts"], Xt), 1e-12, 1))
        np.save(paths.v3 / "logits" / f"{ds}_{task}" / f"seedexp{tag}_s{seed}_test.npy", zt.astype(np.float16))
        for r in res.values():
            r.pop("_experts")
        save_json({"seed": seed, "members": counts, "ens4_rule": base, "configs": res, "chosen": chosen,
                   "test_chosen": classification_metrics(yt, zt, n, data.classes, full=True)}, out_f)
        log.info("s%d chosen on validation: %s -> test %.4f (ens4+rule %.4f)", seed, chosen, res[chosen]["test"], base["test"])


if __name__ == "__main__":
    main()
