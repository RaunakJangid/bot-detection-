"""Reproduction of Almahaqeri et al. 2026 (Sci Rep, doi 10.1038/s41598-026-47399-5), CICIoT2023 34-class XGBoost,
under (a) their protocol and (b) our strict protocol.

Their pipeline (from the paper): stratified sample of 5,000,000 flows (>= 1000 per class); z-score on train; LightGBM
gain-importance feature selection with a median threshold (23 of 46 features, IAT among them); stratified random
80/20 split; XGBoost tuned by grid search (5-fold CV, macro-F1; final values not reported).
(a) THEIR protocol: pool the CICIoT2023 subset (train+validation+test interim files, no cross-split de-dup, IAT kept),
    sample 5M, random stratified 80/20; selection on train; small grid (max_depth {6, 10, 14}) chosen on a 10% slice
    of train (instead of 5-fold CV, for cost); test on the 20%.
(b) STRICT protocol: same pipeline on our strict data (IAT removed, cross-split de-dup, fixed validation/test rows);
    5M stratified training sample; selection on train; grid chosen on validation; test scored once."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.data.preprocess import LABEL, stratified_split
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("almahaqeri")


def select(X, y, names, rng):
    import lightgbm as lgb
    i = stratified_sample(y, min(len(y), 1_000_000), rng, 500)
    m = lgb.LGBMClassifier(n_estimators=200, importance_type="gain", verbose=-1).fit(X[i], y[i])
    imp = m.feature_importances_
    keep = np.flatnonzero(imp >= np.median(imp))
    return keep, [names[k] for k in keep]


def fit_grid(X, y, Xs, ys, n):
    best = None
    for depth in (6, 10, 14):
        c = xgb.XGBClassifier(n_estimators=2000, max_depth=depth, learning_rate=0.1, tree_method="hist", device="cuda",
                              early_stopping_rounds=50, random_state=0, objective="multi:softprob", num_class=n)
        c.fit(X, y, eval_set=[(Xs, ys)], verbose=False)
        f = f1_score(ys, c.predict(Xs), average="macro")
        if best is None or f > best[0]:
            best = (f, depth, c)
    return best


def evaluate(c, X, y):
    p = np.vstack([c.predict_proba(X[i:i + 500_000]) for i in range(0, len(X), 500_000)]).argmax(1)
    return {"macro_f1": float(100 * f1_score(y, p, average="macro")), "accuracy": float(100 * accuracy_score(y, p))}


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    rng = np.random.default_rng(0)
    out = {}
    # (a) their protocol
    I = P.interim("ciciot")
    df = pd.concat([pd.read_parquet(I / f"{s}.parquet") for s in ("train", "validation", "test")], ignore_index=True)
    names = [c for c in df.columns if c != LABEL]
    labels = sorted(df[LABEL].unique()); y_all = df[LABEL].map({l: i for i, l in enumerate(labels)}).to_numpy()
    take = stratified_sample(y_all, 5_000_000, rng, 1000)
    X5 = df[names].to_numpy(np.float32)[take]; y5 = y_all[take]; del df
    X5[~np.isfinite(X5)] = 0
    parts = stratified_split(y5, (0.8, 0.0, 0.2), rng)
    tr, te = parts["train"], parts["test"]
    mu, sd = X5[tr].mean(0), X5[tr].std(0) + 1e-9
    Z = (X5 - mu) / sd
    keep, kn = select(Z[tr], y5[tr], names, rng)
    sl = rng.random(len(tr)) < 0.1
    f, depth, c = fit_grid(Z[tr[~sl]][:, keep], y5[tr[~sl]], Z[tr[sl]][:, keep], y5[tr[sl]], len(labels))
    out["their_protocol"] = {"n_features": len(keep), "features": kn, "iat_selected": "IAT" in kn, "depth": depth,
                             "test": evaluate(c, Z[te][:, keep], y5[te])}
    log.info("(a) their protocol: %d features (IAT %s), depth %d -> %s", len(keep), "IAT" in kn, depth, out["their_protocol"]["test"])
    save_json(out, P.v3 / "ablation" / "almahaqeri_repro.json")
    # (b) strict protocol
    d = load_processed(P.processed("ciciot"), "class34")
    trd, va, ted = d.splits["train"], d.splits["validation"], d.splits["test"]
    it = stratified_sample(trd.y, 5_000_000, rng, 1000)
    X, y = np.asarray(trd.X[it], np.float32), trd.y[it]
    keep, kn = select(X, y, list(d.features), rng)
    iv = stratified_sample(va.y, 500_000, rng, 200)
    f, depth, c = fit_grid(X[:, keep], y, np.asarray(va.X[iv], np.float32)[:, keep], va.y[iv], d.n_classes)
    out["strict_protocol"] = {"n_features": len(keep), "features": kn, "depth": depth,
                              "test": evaluate(c, np.asarray(ted.X, np.float32)[:, keep], ted.y)}
    log.info("(b) strict protocol: %d features, depth %d -> %s", len(keep), depth, out["strict_protocol"]["test"])
    save_json(out, P.v3 / "ablation" / "almahaqeri_repro.json")


if __name__ == "__main__":
    main()
