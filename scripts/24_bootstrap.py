"""Paired bootstrap on the identical test rows: 95% CI of each model's test macro-F1 and of the difference
(reference - model). Predictions are read from outputs_v3/logits (argmax; for ens4+rule the validation-fitted
bias from detector_v4 is applied). Writes outputs_v3/bootstrap/<dataset>_<task>.json."""

import json

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.posthoc import apply_bias
from shield.utils.config import Paths
from shield.utils.io import save_json


def macro_f1(y, p, n):
    conf = np.bincount(y * n + p, minlength=n * n).reshape(n, n).astype(float)
    tp = np.diag(conf)
    present = conf.sum(1) > 0
    return float((2 * tp / np.maximum(conf.sum(0) + conf.sum(1), 1))[present].mean())


def main():
    p = parser(__doc__)
    p.add_argument("--models", nargs="*", required=True)
    p.add_argument("--reference", default="ens4+rule")
    p.add_argument("--B", type=int, default=1000)
    args = p.parse_args()
    paths = Paths(args.smoke)
    ds, task = args.dataset, args.task
    data = load_processed(paths.processed(ds), task, splits=("test",))
    y, n = data.splits["test"].y.astype(np.int64), data.n_classes
    lc = paths.v3 / "logits" / f"{ds}_{task}"
    preds = {}
    for m in set(args.models) | {args.reference}:
        if m == "ens4+rule":
            bias = np.asarray(json.load(open(paths.v3 / "detector_v4" / f"{ds}_{task}.json"))["rule_bias"])
            preds[m] = apply_bias(np.load(lc / "ens4_s0_test.npy").astype(np.float32), bias).argmax(1)
        else:
            preds[m] = np.load(lc / f"{m}_test.npy").astype(np.float32).argmax(1)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.B)]
    ref = preds[args.reference]
    out = {}
    for m, pr in preds.items():
        f = np.array([macro_f1(y[i], pr[i], n) for i in idx])
        r = {"macro_f1": macro_f1(y, pr, n), "ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}
        if m != args.reference:
            fr = np.array([macro_f1(y[i], ref[i], n) for i in idx])
            d = fr - f
            r["diff_vs_ref"] = float(macro_f1(y, ref, n) - r["macro_f1"])
            r["diff_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            r["p_ref_not_better"] = float((d <= 0).mean())
        out[m] = r
        print(m, {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in r.items()}, flush=True)
    save_json(out, paths.v3 / "bootstrap" / f"{ds}_{task}.json")


if __name__ == "__main__":
    main()
