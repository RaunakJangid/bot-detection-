"""Low-volume-traffic specialist (v3, no leakage). The residual macro-F1 loss of the best model sits in rare
low-rate attacks confused with benign/spoofing traffic. A specialist XGBoost is trained only on that group of
classes (floods excluded, class-balanced weights); when the main ensemble (ens4 + rule) predicts a group class,
the group's probability mass is redistributed by the specialist: p'(c) = p_main(G) * p_spec(c | G) for c in G.
The blend weight lam in p'(c) = p_main(G) * [lam * p_spec + (1 - lam) * p_main(c | G)] and its adoption are chosen
on VALIDATION macro-F1; test scored once."""

import json

import numpy as np
import xgboost as xgb

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("specialist")
GROUP = ["BenignTraffic", "DNS_Spoofing", "MITM-ArpSpoofing", "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep",
         "Recon-PortScan", "VulnerabilityScan", "XSS", "SqlInjection", "CommandInjection", "Backdoor_Malware",
         "Uploading_Attack", "BrowserHijacking", "DictionaryBruteForce"]


def main():
    p = parser(__doc__)
    p.add_argument("--max-benign", type=int, default=2_000_000)
    p.add_argument("--auto-group", type=float, default=None,
                   help="group = benign + classes whose VALIDATION F1 (main model) is below this value")
    args = p.parse_args()
    paths = Paths(args.smoke)
    ds, task = args.dataset, args.task
    data = load_processed(paths.processed(ds), task)
    tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
    n = data.n_classes
    lc = paths.v3 / "logits" / f"{ds}_{task}"
    bias = np.asarray(json.load(open(paths.v3 / "detector_v4" / f"{ds}_{task}.json"))["rule_bias"])
    ben_name = next(c for c in data.classes if c in ("BenignTraffic", "Benign", "Normal"))
    if args.auto_group is not None:   # chosen on validation only
        zv = apply_bias(np.load(lc / "ens4_s0_validation.npy").astype(np.float32), bias)
        pc = classification_metrics(va.y, zv, n, data.classes, full=True)["per_class"]
        names = [ben_name] + [c for c in data.classes if c != ben_name and pc[c]["f1"] < args.auto_group]
    else:
        names = GROUP
    G = np.array([data.classes.index(c) for c in names])
    log.info("group: %s", names)
    gmap = np.full(n, -1); gmap[G] = np.arange(len(G))
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(np.isin(tr.y, G))
    ben = data.classes.index(ben_name)
    b = idx[tr.y[idx] == ben]
    if len(b) > args.max_benign:   # keep the training set tractable; benign is far from rare
        idx = np.sort(np.concatenate([idx[tr.y[idx] != ben], rng.choice(b, args.max_benign, replace=False)]))
    ys = gmap[tr.y[idx]]
    cnt = np.bincount(ys, minlength=len(G)).astype(float)
    w = 1 / np.sqrt(cnt); w = w / (w * cnt).sum() * cnt.sum()
    vi = np.flatnonzero(np.isin(va.y, G))
    es = vi[stratified_sample(va.y[vi], min(len(vi), 300_000), rng, 30)]
    log.info("specialist train rows %d (group %d classes), class counts min %d", len(idx), len(G), int(cnt.min()))
    clf = xgb.XGBClassifier(n_estimators=3000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=100, random_state=0,
                            objective="multi:softprob" if len(G) > 2 else "binary:logistic")
    clf.fit(np.asarray(tr.X[idx]), ys, sample_weight=w[ys], eval_set=[(np.asarray(va.X[es]), gmap[va.y[es]])], verbose=False)

    def combine(split, X, lam):
        pm = softmax(apply_bias(np.load(lc / f"ens4_s0_{split}.npy").astype(np.float32), bias).astype(np.float64))
        ps = clf.predict_proba(np.asarray(X))
        mass = pm[:, G].sum(1, keepdims=True)
        cond = pm[:, G] / np.maximum(mass, 1e-12)
        out = pm.copy()
        out[:, G] = mass * (lam * ps + (1 - lam) * cond)
        return np.log(np.clip(out, 1e-12, 1))

    Xv = np.asarray(va.X)
    f1v = {lam: classification_metrics(va.y, combine("validation", Xv, lam), n)["macro_f1"] for lam in (0.0, 0.25, 0.5, 0.75, 1.0)}
    lam = max(f1v, key=f1v.get)
    log.info("validation macro-F1 by blend weight: %s -> chosen %.2f", {k: round(v, 4) for k, v in f1v.items()}, lam)
    zt = combine("test", te.X, lam)
    m = classification_metrics(te.y, zt, n, data.classes, full=True)
    m0 = classification_metrics(te.y, combine("test", te.X, 0.0), n)
    tag = "specialist_auto" if args.auto_group is not None else "specialist"
    np.save(lc / f"{tag}_s0_test.npy", zt.astype(np.float16))
    save_json({"group": names, "lambda": lam, "val_by_lambda": f1v, "test": m, "test_without": m0["macro_f1"]}, paths.v3 / "specialist" / f"{ds}_{task}_{tag}.json")
    log.info("%s %s: test macro-F1 %.4f with specialist (lam %.2f) vs %.4f without", ds, task, m["macro_f1"], lam, m0["macro_f1"])


if __name__ == "__main__":
    main()
