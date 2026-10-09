"""How much would 9x more training data help? XGBoost (v2 settings) on stratified fractions of the CICIoT
34-class training split; VALIDATION macro-F1 only. Fit F(n) = a - b * n^(-c) and extrapolate to 9x."""
import json

import numpy as np
import xgboost as xgb
from scipy.optimize import curve_fit
from sklearn.metrics import f1_score

from shield.data.datasets import load_processed, stratified_sample
from shield.utils.config import Paths

paths = Paths()
data = load_processed(paths.processed("ciciot"), "class34")
tr, va = data.splits["train"], data.splits["validation"]
Xv, yv = np.asarray(va.X), va.y
rare = [data.classes.index(c) for c in ["Uploading_Attack", "Recon-PingSweep", "Backdoor_Malware", "XSS",
                                        "SqlInjection", "CommandInjection", "BrowserHijacking", "DictionaryBruteForce"]]
res = []
for frac in (0.125, 0.25, 0.5, 1.0):
    idx = (np.arange(len(tr.y)) if frac == 1 else
           np.sort(np.concatenate([np.random.default_rng(0).choice(np.flatnonzero(tr.y == c),
                                   max(1, int(round(frac * (tr.y == c).sum()))), replace=False)
                                   for c in np.unique(tr.y)])))
    clf = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=50, objective="multi:softprob", random_state=0)
    vi = stratified_sample(yv, 300_000, np.random.default_rng(1), 200)   # early-stopping subset of validation
    clf.fit(np.asarray(tr.X[idx]), tr.y[idx], eval_set=[(Xv[vi], yv[vi])], verbose=False)
    pred = clf.predict(Xv)
    pc = f1_score(yv, pred, average=None, labels=range(data.n_classes))
    r = {"frac": frac, "n": int(len(idx)), "macro_f1": float(pc.mean()), "rare_f1": float(pc[rare].mean()),
         "common_f1": float(np.delete(pc, rare).mean())}
    res.append(r)
    print(r, flush=True)

n = np.array([r["n"] for r in res], float)
law = lambda x, a, b, c: a - b * x ** (-c)
out = {"points": res}
for key in ("macro_f1", "rare_f1", "common_f1"):
    y = np.array([r[key] for r in res])
    try:
        (a, b, c), _ = curve_fit(law, n / n[-1], y, p0=[y[-1] + 0.02, 0.02, 0.5], maxfev=20000,
                                 bounds=([0, 0, 0.01], [1, 1, 3]))
        out[key] = {"now": float(y[-1]), "at_9x": float(law(9.0, a, b, c)), "asymptote": float(a), "c": float(c)}
    except RuntimeError:
        out[key] = {"now": float(y[-1]), "fit": "failed"}
    print(key, out[key], flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
