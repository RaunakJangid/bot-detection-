"""Calibration choices for IntMLP, scored on VALIDATION macro-F1 (the choice is made on validation)."""
import numpy as np
import torch
from shield.data.datasets import load_processed, predict_logits, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.predictions import load_student, run_dir
from shield.models.quant import IntMLP, calibrate
from shield.utils.config import Paths

paths = Paths()
cpu = torch.device("cpu")


def balanced(y, per_class, rng):
    idx = [rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()), replace=False) for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


for ds, task, var, seed in [("ciciot", "binary", "kd_all", 2), ("ciciot", "family", "scratch_all", 0),
                            ("ciciot", "family", "shield", 0), ("ciciot", "class34", "shield", 0),
                            ("botiot", "category", "shield", 0)]:
    data = load_processed(paths.processed(ds), task)
    tr, va = data.splits["train"], data.splits["validation"]
    model, S = load_student(run_dir(paths, ds, task, var, seed), cpu)
    cols = lambda X: np.asarray(X)[:, S] if len(S) != X.shape[1] else np.asarray(X)
    vi = np.sort(np.random.default_rng(0).choice(len(va.y), min(len(va.y), 400_000), replace=False))
    Xv, yv = cols(va.X[vi]), va.y[vi]
    f1 = lambda z: classification_metrics(yv, z, data.n_classes)["macro_f1"]
    rng = np.random.default_rng(seed)
    cal = {"prop": cols(tr.X[stratified_sample(tr.y, 200000, rng, 50)]),
           "bal": cols(tr.X[balanced(tr.y, 20000, rng)])}
    print(f"== {ds}/{task} {var} s{seed}: fp32 {f1(predict_logits(model, Xv, cpu)):.4f}")
    for bits in (8, 16):
        res = []
        for cname, Xc in cal.items():
            for ip in (99.99, 100.0):
                for ap in (99.99, 100.0):
                    q = IntMLP(model, calibrate(model, Xc, ip, ap), act_bits=bits)
                    res.append(f"{cname} in{ip:g}/act{ap:g} {f1(q.forward(Xv)):.4f}")
        print(f"  W8A{bits}: " + " | ".join(res))
