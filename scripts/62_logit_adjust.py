"""Follow-up to row 28z: keep the class-weight gain without its extra benign false alarms (logit adjustment).

Class weights count^-0.5 train XGBoost as if the class prior were pi^0.5 (normalised), so it over-predicts rare
classes and raises benign false alarms (3.40% -> 5.97% on the holdout). Post-hoc logit adjustment (Menon et al.,
ICLR 2021) moves the decision back toward the true prior: log p_adj = log p + tau * 0.5 * log pi, tau in [0, 1]
(tau = 1 undoes the weighting fully). Data: ciciot_fullho 34-class, the same 10M training rows and recipe as row 28z.
tau chosen on validation two ways: (a) best validation macro-F1; (b) best validation macro-F1 with benign false alarms
no higher than the unweighted base model's. Holdout scored once for base, weights (tau = 0) and each chosen tau."""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("logit_adjust")
_s = importlib.util.spec_from_file_location("rb", Path(__file__).with_name("50_rebalance.py"))
rb = importlib.util.module_from_spec(_s); _s.loader.exec_module(rb)
TAUS = np.round(np.arange(0, 1.01, 0.1), 2)


def proba(clf, X):
    return np.vstack([clf.predict_proba(X[i:i + 500_000]) for i in range(0, len(X), 500_000)])


def main():
    parser(__doc__, datasets=False).parse_args()
    P = Paths(False)
    d = load_processed(P.processed("ciciot_fullho"), "class34")
    tr, va, te = d.splits["train"], d.splits["validation"], d.splits["test"]
    n, cls = d.n_classes, d.classes
    ben = cls.index("BenignTraffic")
    counts = np.bincount(tr.y, minlength=n).astype(float)
    pi = counts / counts.sum()
    rng = np.random.default_rng(0)                       # same draws as 50_rebalance.py
    iv = stratified_sample(va.y, 500_000, rng, 200)
    Xe, ye = np.asarray(va.X[iv], np.float32), va.y[iv]
    ib = stratified_sample(tr.y, 10_000_000, rng, 1000)
    X, y = np.asarray(tr.X[ib], np.float32), tr.y[ib]
    w = counts ** -0.5; w = w / w[y].mean()
    Xv, yv, Xt, yt = np.asarray(va.X, np.float32), va.y, np.asarray(te.X, np.float32), te.y

    def score(pred, yy):
        return {"macro_f1": float(100 * f1_score(yy, pred, labels=np.arange(n), average="macro", zero_division=0)),
                "benign_false_alarm": float(100 * (pred[yy == ben] != ben).mean())}

    base = rb.fit(X, y, None, Xe, ye, n)
    pb_v, pb_t = proba(base, Xv), proba(base, Xt)
    res = {"base": {"validation": score(pb_v.argmax(1), yv), "holdout": score(pb_t.argmax(1), yt)}}
    log.info("base: %s", res["base"])
    del base
    wm = rb.fit(X, y, w[y], Xe, ye, n)
    lv, lt = np.log(np.clip(proba(wm, Xv), 1e-12, 1)), np.log(np.clip(proba(wm, Xt), 1e-12, 1))
    adj = 0.5 * np.log(pi)
    res["tau_grid_validation"] = {}
    for tau in TAUS:
        res["tau_grid_validation"][str(tau)] = score((lv + tau * adj).argmax(1), yv)
        log.info("tau %.1f: validation %s", tau, res["tau_grid_validation"][str(tau)])
    g = res["tau_grid_validation"]
    tau_a = max(TAUS, key=lambda t: g[str(t)]["macro_f1"])
    ok = [t for t in TAUS if g[str(t)]["benign_false_alarm"] <= res["base"]["validation"]["benign_false_alarm"]]
    tau_b = max(ok, key=lambda t: g[str(t)]["macro_f1"]) if ok else None
    res["chosen"] = {"best_f1": float(tau_a), "best_f1_fa_capped": None if tau_b is None else float(tau_b)}
    res["holdout"] = {f"tau_{t}": score((lt + t * adj).argmax(1), yt) for t in sorted({0.0, tau_a} | ({tau_b} if tau_b is not None else set()))}
    log.info("chosen %s | holdout %s | base holdout %s", res["chosen"], res["holdout"], res["base"]["holdout"])
    save_json(res, P.v3 / "rebalance" / "logit_adjust.json")


if __name__ == "__main__":
    main()
