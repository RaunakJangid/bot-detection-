"""Distribution shift between captures and operating-point transfer (no new training, cached logits).

Design (after TESSERACT, Pendlebury et al. USENIX Sec 2019, and Arp et al. USENIX Sec 2022):
  * CICIoMT2024 ships capture-level train/test files; our validation is a slice of the TRAIN captures, so
    validation -> test is a genuine change of capture. CICIoT2023 validation/test are random slices of the same
    captures and serve as the no-shift CONTROL: any val->test gap there is sampling noise.
  * Per model: macro-F1 on validation vs test; the alarm threshold on p(attack) is fixed on validation benign
    flows for a target FPR (1%, 0.1%) and the REALISED test FPR / recall are reported (does the operating point
    survive a change of capture?).
  * Base-rate view: precision and false alerts per million flows at attack prevalence 50%, 10%, 1%, 0.1%,
    computed from the realised test TPR/FPR (precision = TPR*pi / (TPR*pi + FPR*(1-pi))).
  * Shift diagnosis: a LightGBM domain classifier separates validation from test BENIGN flows (balanced
    subsample, 2-fold); AUC ~0.5 = no shift; per-feature shift = Kolmogorov-Smirnov statistic."""

import json

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("shift")
JOBS = [("ciciomt", "category"), ("ciciomt", "attack"), ("ciciomt", "binary"),
        ("ciciot", "class34"), ("ciciot", "family"), ("ciciot", "binary"), ("ciciot_full", "class34")]
PREV = (0.5, 0.1, 0.01, 0.001)


def main():
    p = parser(__doc__, datasets=False)
    args = p.parse_args()
    import lightgbm as lgb
    P = Paths(args.smoke)
    out = {}
    for ds, task in JOBS:
        lc = P.v3 / "logits" / f"{ds}_{task}"
        if not lc.exists():
            continue
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        n = d.n_classes
        yv, yt = d.splits["validation"].y, d.splits["test"].y
        ben = next(i for i, c in enumerate(d.classes) if c in ("Benign", "BenignTraffic", "Normal"))
        av, at = yv != ben, yt != ben
        rule = P.v3 / "detector_v4" / f"{ds}_{task}.json"
        bias = np.asarray(json.load(open(rule))["rule_bias"]) if rule.exists() else None
        models = sorted({f.name[:-len("_validation.npy")] for f in lc.glob("*_validation.npy")
                         if (lc / f.name.replace("_validation", "_test")).exists()})
        res = {}
        for m in models:
            zv = np.load(lc / f"{m}_validation.npy").astype(np.float32)
            zt = np.load(lc / f"{m}_test.npy").astype(np.float32)
            if m == "ens4_s0" and bias is not None:
                zv, zt = apply_bias(zv, bias), apply_bias(zt, bias)
            r = {"val_macro_f1": classification_metrics(yv, zv, n)["macro_f1"],
                 "test_macro_f1": classification_metrics(yt, zt, n)["macro_f1"]}
            sv = 1 - softmax(zv.astype(np.float64))[:, ben]
            st = 1 - softmax(zt.astype(np.float64))[:, ben]
            for target in (0.01, 0.001):
                thr = np.quantile(sv[~av], 1 - target)
                tpr, fpr = float((st[at] > thr).mean()), float((st[~at] > thr).mean())
                r[f"fpr@{target}"] = {"val_fpr": float((sv[~av] > thr).mean()), "test_fpr": fpr, "test_recall": tpr,
                                      "precision_at_prevalence": {str(pi): tpr * pi / max(tpr * pi + fpr * (1 - pi), 1e-12)
                                                                  for pi in PREV},
                                      "false_alerts_per_1M_flows": {str(pi): fpr * (1 - pi) * 1e6 for pi in PREV}}
            res[m] = r
        # shift diagnosis on benign flows
        Xv = np.asarray(d.splits["validation"].X[np.flatnonzero(~av)])
        Xt = np.asarray(d.splits["test"].X[np.flatnonzero(~at)])
        rng = np.random.default_rng(0)
        k = min(len(Xv), len(Xt), 50_000)
        A = np.vstack([Xv[rng.choice(len(Xv), k, replace=False)], Xt[rng.choice(len(Xt), k, replace=False)]])
        y = np.r_[np.zeros(k), np.ones(k)]
        fold = rng.integers(0, 2, len(y)); s = np.zeros(len(y))
        for f in (0, 1):
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1).fit(A[fold != f], y[fold != f])
            s[fold == f] = clf.predict_proba(A[fold == f])[:, 1]
        ks = {d.features[j]: float(ks_2samp(A[:k, j], A[k:, j]).statistic) for j in range(A.shape[1])}
        out[f"{ds}_{task}"] = {"models": res, "benign_domain_auc": float(roc_auc_score(y, s)),
                               "top_shifted_features": dict(sorted(ks.items(), key=lambda t: -t[1])[:8]),
                               "n_val_benign": int((~av).sum()), "n_test_benign": int((~at).sum())}
        log.info("%s_%s: benign val-vs-test domain AUC %.3f | %s", ds, task, out[f"{ds}_{task}"]["benign_domain_auc"],
                 {m: (round(100 * v["val_macro_f1"], 1), round(100 * v["test_macro_f1"], 1),
                      round(100 * v["fpr@0.01"]["test_fpr"], 2)) for m, v in res.items()})
        save_json(out, P.v3 / "shift" / "shift_base_rate.json")


if __name__ == "__main__":
    main()
