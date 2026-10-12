"""Window-size shortcut in CICIoT2023 (Neto et al. 2023, Sec. 4: benign, web, recon, spoofing and brute force rows
average windows of 10 packets; DDoS, DoS and Mirai rows average windows of 100 packets).

If window-count features carry the window size, a model can separate flood families from the rest without
learning traffic behaviour. Tests on the strict CICIoT2023 subset (IAT removed, de-duplicated; validation for early
stopping, test scored once), XGBoost, 2M-row stratified training sample, same settings in every arm:
  (1) per-class medians of the candidate window features (raw values, interim data);
  (2) window features only (Number, Weight) -> how much of each task they solve alone;
  (3) all strict features vs all minus the window features -> how much accuracy depends on them."""

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("window")
WIN = ["Number", "Weight"]


def fit_eval(d, cols, seed=0, cap=2_000_000):
    tr, va, te = d.splits["train"], d.splits["validation"], d.splits["test"]
    ix = [d.features.index(c) for c in cols]
    rng = np.random.default_rng(seed)
    it = stratified_sample(tr.y, min(cap, len(tr.y)), rng, 1000)
    iv = stratified_sample(va.y, min(500_000, len(va.y)), rng, 200)
    clf = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=50, random_state=seed)
    clf.fit(np.asarray(tr.X[it])[:, ix], tr.y[it], eval_set=[(np.asarray(va.X[iv])[:, ix], va.y[iv])], verbose=False)
    Xt = np.asarray(te.X)[:, ix]
    pr = np.vstack([clf.predict_proba(Xt[i:i + 500_000]) for i in range(0, len(Xt), 500_000)])
    m = classification_metrics(te.y, np.log(np.clip(pr, 1e-12, 1)), d.n_classes)
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    out = {}
    raw = pd.read_parquet(P.interim("ciciot") / "validation.parquet")
    lab = [c for c in raw.columns if c.lower() == "label"][0]
    med = raw.groupby(lab)[WIN + ["Tot sum", "Magnitue"]].median()
    out["per_class_median"] = {k: {c: float(v) for c, v in r.items()} for k, r in med.iterrows()}
    log.info("per-class medians:\n%s", med.round(2).to_string())
    for task in ("binary", "family", "class34"):
        d = load_processed(P.processed("ciciot"), task)
        allc = list(d.features)
        res = {"window_only": fit_eval(d, WIN), "all": fit_eval(d, allc),
               "all_minus_window": fit_eval(d, [c for c in allc if c not in WIN])}
        out[task] = res
        log.info("%s: %s", task, {k: (round(100 * v["macro_f1"], 2), round(100 * v["accuracy"], 2)) for k, v in res.items()})
        save_json(out, P.v3 / "ablation" / "window_shortcut.json")


if __name__ == "__main__":
    main()
