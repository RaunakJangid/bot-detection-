"""E1: can the 1% FPR operating point be restored after a change of capture, with few or no labels? (cached logits)

Setting (fixed before the run, research/audit/EXPERIMENT_DECISION_PLAN.md):
  * Score = p(attack) = 1 - p(benign), as in 34_shift_and_base_rate.py; target FPR alpha = 1%.
  * After deployment the detector sees a burn-in window of K flows from the NEW captures (test split). The window
    contains attacks at prevalence pi (unknown to the methods). Burn-in rows are excluded from evaluation.
  * Methods:
      fixed     threshold from validation benign flows (current practice)
      clean     conformal quantile of K benign burn-in flows (needs a clean, attack-free burn-in; no labels otherwise)
      naive     same quantile on the contaminated window (no labels, ignores attacks)
      corrected label-free: pi_hat = share of window above the validation 0.1%-FPR threshold; threshold = window
                quantile at 1 - (pi_hat + alpha * (1 - pi_hat))
  * Primary setting: K = 2000, pi = 1%. Grid: K in {500, 1000, 2000, 5000}, pi in {0, 1%, 10%}; 20 random draws.
  * CICIoMT (validation = train captures, test = new captures) is the shift case; CICIoT (random split) is the
    control, where recalibration must not hurt.
  * Decision rule: a method is kept if, at the primary setting, the median test FPR over models is <= 1.5% with
    median recall loss <= 3 points vs fixed, on >= 2 of 3 CICIoMT tasks and for >= 3 model families.
Limitation: test rows carry no timestamps, so burn-in flows are a random sample of the new captures, not the first
K flows in time."""

import re

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("recalib")
JOBS = [("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack"),
        ("ciciot", "binary"), ("ciciot", "family"), ("ciciot", "class34")]
ALPHA, KS, PIS, REPS = 0.01, (500, 1000, 2000, 5000), (0.0, 0.01, 0.1), 20


def family(m):
    b = re.sub(r"_s\d+$", "", m)
    if b in ("xgboost", "xgbsub", "catboost", "lgbm", "lgbm_tuned", "two_stage"):
        return "gbdt"
    if b in ("ensemble", "ens4", "treeens"):
        return "ensemble"
    if b == "scratch_all" or b == "mlp_s":
        return "small_mlp_scratch"
    if b.startswith(("kd_", "shield", "v3-")):
        return "small_mlp_kd"
    return "large_neural"


def conformal_q(s, alpha):
    """Threshold t with P(benign score > t) <= alpha under exchangeability (split-conformal order statistic)."""
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def above(sorted_s, t):
    return len(sorted_s) - np.searchsorted(sorted_s, t, side="right")


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    out = {}
    for ds, task in JOBS:
        lc = P.v3 / "logits" / f"{ds}_{task}"
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        ben = next(i for i, c in enumerate(d.classes) if c in ("Benign", "BenignTraffic", "Normal"))
        bv, bt = d.splits["validation"].y == ben, d.splits["test"].y == ben
        models = sorted({f.name[:-len("_validation.npy")] for f in lc.glob("*_validation.npy")
                         if (lc / f.name.replace("_validation", "_test")).exists()})
        res = {}
        for m in models:
            sv = 1 - softmax(np.load(lc / f"{m}_validation.npy").astype(np.float64))[:, ben]
            st = 1 - softmax(np.load(lc / f"{m}_test.npy").astype(np.float64))[:, ben]
            sb, sa = st[bt], st[~bt]
            sb_sorted, sa_sorted = np.sort(sb), np.sort(sa)
            t_fix = np.quantile(sv[bv], 1 - ALPHA)
            t_hi = np.quantile(sv[bv], 1 - ALPHA / 10)
            rec = lambda t: above(sa_sorted, t) / len(sa)
            r = {"family": family(m), "fixed": {"fpr": above(sb_sorted, t_fix) / len(sb), "recall": rec(t_fix)}}
            rng = np.random.default_rng(0)
            for K in KS:
                for pi in PIS:
                    acc = {k: [] for k in ("clean", "naive", "corrected")}
                    for _ in range(REPS):
                        n_att = int(round(K * pi))
                        ib = rng.choice(len(sb), K - n_att, replace=False)
                        wb, wa = sb[ib], sa[rng.choice(len(sa), n_att, replace=False)]
                        win = np.r_[wb, wa]
                        pi_hat = float((win > t_hi).mean())
                        q = min(pi_hat + ALPHA * (1 - pi_hat), 0.999)
                        thr = {"clean": conformal_q(sb[rng.choice(len(sb), K, replace=False)], ALPHA),
                               "naive": conformal_q(win, ALPHA),
                               "corrected": np.quantile(win, 1 - q)}
                        keep = np.ones(len(sb), bool); keep[ib] = False     # evaluate on benign flows not in burn-in
                        ev = np.sort(sb[keep])
                        for k, t in thr.items():
                            acc[k].append((above(ev, t) / len(ev), rec(t)))
                    for k, v in acc.items():
                        a = np.asarray(v)
                        r[f"{k}_K{K}_pi{pi}"] = {"fpr_median": float(np.median(a[:, 0])),
                                                 "fpr_p90": float(np.quantile(a[:, 0], 0.9)),
                                                 "violation_rate": float((a[:, 0] > 1.5 * ALPHA).mean()),
                                                 "recall_median": float(np.median(a[:, 1]))}
            res[m] = r
        out[f"{ds}_{task}"] = res
        prim = lambda k: np.median([v[f"{k}_K2000_pi0.01"]["fpr_median"] for v in res.values()])
        log.info("%s_%s: %d models | median test FPR fixed %.2f%% | clean %.2f%% | naive %.2f%% | corrected %.2f%%",
                 ds, task, len(res), 100 * np.median([v["fixed"]["fpr"] for v in res.values()]),
                 100 * prim("clean"), 100 * prim("naive"), 100 * prim("corrected"))
        save_json(out, P.v3 / "recalib" / "recalib.json")


if __name__ == "__main__":
    main()
