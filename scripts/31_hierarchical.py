"""Two-level (family -> class) decision, no new training, no leakage. The 34-class ensemble's probabilities are
re-factorised as p'(c) = p_fam(F)^a * p34(F)^(1-a) * p34(c | F), where p_fam comes from the separately trained 8-family
ensemble (same rows, family labels) and p34(c | F) is the 34-class ensemble renormalised inside each family.
The family exponent a and a per-class bias (the validation-fitted rule) are chosen on VALIDATION half A,
checked on half B; test scored once. Rationale: rare classes then compete only with their own family, not with
the large DoS/DDoS families."""

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("hier")


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--dataset", default="ciciot_full")
    p.add_argument("--model", default="ens4_s0")
    args = p.parse_args()
    P, ds = Paths(args.smoke), args.dataset
    d34, dfam = load_processed(P.processed(ds), "class34"), load_processed(P.processed(ds), "family")
    tr34, trf = d34.splits["train"].y, dfam.splits["train"].y
    fam_of = np.array([np.bincount(trf[tr34 == c], minlength=dfam.n_classes).argmax() for c in range(d34.n_classes)])
    M = np.eye(dfam.n_classes)[fam_of]                                        # (34, n_fam)
    load = lambda t, s: np.load(P.v3 / "logits" / f"{ds}_{t}" / f"{args.model}_{s}.npy").astype(np.float64)
    yv, yt, n = d34.splits["validation"].y, d34.splits["test"].y, d34.n_classes
    prior = np.bincount(tr34, minlength=n) / len(tr34)

    def combine(split, a):
        p34, pf = softmax(load("class34", split)), softmax(load("family", split))
        mass = p34 @ M                                                        # p34(F)
        cond = p34 / np.maximum(mass @ M.T, 1e-12)                            # p34(c | F(c))
        fam = a * np.log(np.clip(pf @ M.T, 1e-12, 1)) + (1 - a) * np.log(np.clip(mass @ M.T, 1e-12, 1))
        return fam + np.log(np.clip(cond, 1e-12, 1))      # a = 0 is exactly the flat 34-class model

    A = np.zeros(len(yv), bool); A[stratified_sample(yv, len(yv) // 2, np.random.default_rng(0), 1)] = True
    f1 = lambda y, z: classification_metrics(y, z, n)["macro_f1"]
    res = {}
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        zv = combine("validation", a)
        bias = fit_decision_rule(zv[A], yv[A], prior)["bias"]
        res[a] = {"valA": f1(yv[A], apply_bias(zv[A], bias)), "valB": f1(yv[~A], apply_bias(zv[~A], bias)), "bias": bias}
        log.info("a=%.1f valA %.4f valB %.4f", a, res[a]["valA"], res[a]["valB"])
    a = max(res, key=lambda k: res[k]["valA"])
    zt = apply_bias(combine("test", a), res[a]["bias"])
    flat = apply_bias(combine("test", 0.0), res[0.0]["bias"])
    out = {"chosen_a": a, "by_a": {k: {kk: v for kk, v in r.items() if kk != "bias"} for k, r in res.items()},
           "test_hier": classification_metrics(yt, zt, n, d34.classes, full=True), "test_flat_rule": f1(yt, flat)}
    np.save(P.v3 / "logits" / f"{ds}_class34" / f"hier_{args.model}_test.npy", zt.astype(np.float16))
    save_json(out, P.v3 / "hierarchical" / f"{ds}_{args.model}.json")
    log.info("chosen a=%.1f: test %.4f (flat + same rule fit %.4f)", a, out["test_hier"]["macro_f1"], out["test_flat_rule"])


if __name__ == "__main__":
    main()
