"""Screen (go/no-go, not a verdict): can a LOCAL MEMORY of the new network's benign traffic repair lost ranking?

Setting (IoT-23, ranking-loss regime; per-flow scores + features from 41_iot23_shift.py; time order per capture):
  * commissioning window = the first K benign flows of the capture in time (clean commissioning, the easy case);
    its first half builds the memory, its second half calibrates the threshold (no point is scored against itself);
  * memory variants: kNN = mean distance to the 5 nearest memory flows; CENT = distance to the nearest of 64 k-means
    centroids (a few KB, edge-sized). Features standardised with train-capture (validation) benign statistics;
  * fused score (primary, fixed in advance) = mean of the two commissioning-percentiles: detector score and local
    novelty (distance). Also reported, not used for the decision: novelty alone (Kitsune-like) and AND (product);
  * threshold = conformal 1% quantile of the fused score on the calibration half; evaluated on all later flows.
Decision rule (fixed before the run): GO if on the same-family design, for the median model, ROC-AUC >= 0.85 AND
FPR <= 1.5% AND recall loss vs the fixed validation threshold <= 5 points (either memory variant). Else: clear fail."""

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("commission")
ALPHA = 0.01


def conformal(s, a=ALPHA):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def pct(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    P = Paths(args.smoke)
    out = {}
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz")
        models, sv, st, yv, yt = list(Z["models"]), Z["sv"], Z["st"], Z["yv"], Z["yt"]
        cap, ts, Xv, Xt = Z["capture"], Z["ts"], Z["Xv"], Z["Xt"]
        mu, sd = Xv[yv == 0].mean(0), Xv[yv == 0].std(0) + 1e-6
        Xs = (Xt - mu) / sd
        caps = [c for c in np.unique(cap) if ((cap == c) & (yt == 0)).sum() >= args.k + 500]
        res = {"captures": [str(c) for c in caps], "models": {}}
        # memory and novelty depend only on features (shared by all models): compute once per capture
        per_cap = {}
        for c in caps:
            idx = np.flatnonzero(cap == c); idx = idx[np.argsort(ts[idx], kind="stable")]
            ben = np.flatnonzero(yt[idx] == 0)[:args.k]
            mem, cal = idx[ben[: args.k // 2]], idx[ben[args.k // 2:]]
            ev = np.setdiff1d(idx, idx[ben], assume_unique=False)
            nn = NearestNeighbors(n_neighbors=5).fit(Xs[mem])
            km = KMeans(64, n_init=3, random_state=0).fit(Xs[mem])
            nov = {"knn": lambda I: nn.kneighbors(Xs[I])[0].mean(1),
                   "cent": lambda I: km.transform(Xs[I]).min(1)}
            per_cap[c] = {"cal": cal, "ev": ev, "nov": {k: (f(cal), f(ev)) for k, f in nov.items()}}
        for mi, m in enumerate(models):
            t_fix = float(np.quantile(sv[mi][yv == 0], 1 - ALPHA))
            rows = {}
            for c in caps:
                pc = per_cap[c]; cal, ev = pc["cal"], pc["ev"]; y = yt[ev]
                if len(np.unique(y)) < 2:
                    continue
                s_cal, s_ev = st[mi][cal], st[mi][ev]
                r = {"base_roc": roc_auc_score(y, s_ev),
                     "fixed": {"fpr": float((s_ev[y == 0] > t_fix).mean()), "recall": float((s_ev[y == 1] > t_fix).mean())}}
                for vk, (d_cal, d_ev) in pc["nov"].items():
                    ps_ev, pd_ev = pct(s_cal, s_ev), pct(d_cal, d_ev)
                    ps_cal = pct(s_cal, s_cal); pd_cal = pct(d_cal, d_cal)
                    for fk, (f_cal, f_ev) in {"mean": ((ps_cal + pd_cal) / 2, (ps_ev + pd_ev) / 2),
                                              "novelty": (pd_cal, pd_ev), "and": (ps_cal * pd_cal, ps_ev * pd_ev)}.items():
                        thr = conformal(f_cal)
                        r[f"{vk}_{fk}"] = {"roc": roc_auc_score(y, f_ev), "fpr": float((f_ev[y == 0] > thr).mean()),
                                           "recall": float((f_ev[y == 1] > thr).mean())}
                rows[str(c)] = r
            res["models"][m] = rows
        # summary over models (median of per-capture means)
        summ = {}
        keys = [k for k in next(iter(next(iter(res["models"].values())).values())).keys() if k not in ("base_roc",)]
        for k in ["base_roc"] + keys:
            if k == "base_roc":
                summ[k] = float(np.median([np.mean([r[k] for r in rows.values()]) for rows in res["models"].values()]))
                continue
            roc = [np.mean([r[k]["roc"] for r in rows.values()]) for rows in res["models"].values()] if k != "fixed" else None
            fpr = [np.mean([r[k]["fpr"] for r in rows.values()]) for rows in res["models"].values()]
            rec = [np.mean([r[k]["recall"] for r in rows.values()]) for rows in res["models"].values()]
            rec0 = [np.mean([r["fixed"]["recall"] for r in rows.values()]) for rows in res["models"].values()]
            summ[k] = {"roc": float(np.median(roc)) if roc else None, "fpr": float(np.median(fpr)),
                       "recall": float(np.median(rec)), "recall_loss": float(np.median(np.array(rec0) - np.array(rec)))}
        res["summary"] = summ
        out[design] = res
        log.info("%s captures %s | base ROC %.3f | %s", design, res["captures"], summ["base_roc"],
                 {k: (None if v["roc"] is None else round(v["roc"], 3), round(100 * v["fpr"], 2), round(100 * v["recall"], 1),
                      round(100 * v["recall_loss"], 1)) for k, v in summ.items() if k != "base_roc"})
    go = any(out["same_family"]["summary"][f"{v}_mean"]["roc"] >= 0.85 and out["same_family"]["summary"][f"{v}_mean"]["fpr"] <= 0.015
             and out["same_family"]["summary"][f"{v}_mean"]["recall_loss"] <= 0.05 for v in ("knn", "cent"))
    out["decision"] = "GO" if go else "FAIL (pre-set rule not met)"
    log.info("decision (same-family, primary fusion): %s", out["decision"])
    save_json(out, P.v3 / "iot23" / f"commission_screen{args.tag}.json")


if __name__ == "__main__":
    main()
