"""P2 v4, go/no-go test of the "traffic-conditioned escalation" hypothesis (external review): is escalation correlated
with flow rate over time, and does a global escalation share misestimate per-controller expert demand?

Data: time-ordered IoT-23 test captures of both designs (scripts/41 scores, student small_mlp_kd_s0-4), each capture
treated as one traffic source. Rates use the sampler weights w (true class mix: the per-label caps of the sampler
distort raw counts). Escalation: student entropy above the validation 90th percentile (planned 10%).
Per source and time window (60 s and 300 s, relative to the capture start): lambda = weighted flows per second,
q = weighted escalated share.
Tests:
  T-a temporal: per source, Spearman correlation of lambda and q over windows; independence error of expert demand
      E[lambda q] / (E[lambda] E[q]) - 1 (positive = independence underestimates demand);
  T-b aggregation: an emulated network (captures aligned by time since start; sources were recorded separately, so
      co-occurrence is synthetic): 200 random assignments of sources to 3 controllers; per controller and window, actual
      expert demand D = sum lambda_i q_i against (i) global nominal e = 10%, (ii) global measured share of that window;
      relative error and the share of controller-windows where D exceeds the prediction by more than 20%;
  T-c with the per-controller quota (at most 10% of the controller's flows escalated): D_q = min(D, 0.1 sum lambda_i);
      same errors.
Output: outputs_v5/p2/traffic_escalation.json."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
rng = np.random.default_rng(103)


def ent2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    P = Paths(False); res = {"sources": {}, "aggregation": {}}
    srcs = {}
    for design in ("capture_shift", "same_family"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        for cap in np.unique(Z["capture"]):
            m = Z["capture"] == cap
            ts = Z["ts"][m].astype(float); w = Z["w"][m].astype(float) if "w" in Z.files else np.ones(m.sum())
            if len(ts) < 500:
                continue
            esc = []
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                tv = np.quantile(ent2(Z["sv"][i]), 0.9); esc.append(ent2(Z["st"][i][m]) > tv)
            srcs[f"{design}|{cap}"] = (ts - ts.min(), w, np.array(esc))
    for win in (60, 300):
        per, arr = {}, {}
        for name, (t, w, E) in srcs.items():
            b = (t // win).astype(int); nb = b.max() + 1
            lam = np.bincount(b, weights=w, minlength=nb) / win
            q = np.array([np.bincount(b, weights=w * E[s], minlength=nb) / np.maximum(np.bincount(b, weights=w, minlength=nb), 1e-12)
                          for s in range(5)])
            ok = lam > 0
            row = {"windows": int(ok.sum())}
            if ok.sum() >= 5:
                rhos = [spearmanr(lam[ok], q[s][ok]).correlation for s in range(5)]
                ind = [float(np.mean(lam[ok] * q[s][ok]) / (np.mean(lam[ok]) * np.mean(q[s][ok])) - 1) if np.mean(q[s][ok]) > 0 else None
                       for s in range(5)]
                row.update(spearman_median=float(np.nanmedian(rhos)), spearman_range=[float(np.nanmin(rhos)), float(np.nanmax(rhos))],
                           independence_error_median=float(np.median([x for x in ind if x is not None])) if any(x is not None for x in ind) else None,
                           q_median=float(np.median(q[:, ok])), q_p95=float(np.quantile(q[:, ok], .95)))
            per[name] = row; arr[name] = (lam, q)
        res["sources"][str(win)] = per
        # aggregation: emulate a network of these sources on a common relative timeline
        names = [n for n in srcs if per[n]["windows"] >= 5]
        T = max(len(arr[n][0]) for n in names)
        L = np.zeros((len(names), T)); Qs = np.zeros((5, len(names), T))
        for j, n in enumerate(names):
            lam, q = arr[n]; L[j, :len(lam)] = lam; Qs[:, j, :len(lam)] = q
        errs = {k: [] for k in ("nominal", "window_global", "nominal_quota", "window_global_quota")}
        exceed = {k: [] for k in errs}
        for _ in range(200):
            a = rng.integers(0, 3, len(names)); s = rng.integers(0, 5)
            D_all = (L * Qs[s]); e_win = D_all.sum(0) / np.maximum(L.sum(0), 1e-12)
            for c in range(3):
                mc = a == c
                if not mc.any():
                    continue
                lam_c = L[mc].sum(0); D = D_all[mc].sum(0); act = lam_c > 0
                Dq = np.minimum(D, 0.10 * lam_c)
                for key, actual, pred in (("nominal", D, 0.10 * lam_c), ("window_global", D, e_win * lam_c),
                                          ("nominal_quota", Dq, 0.10 * lam_c), ("window_global_quota", Dq, e_win * lam_c)):
                    r = (actual[act] - pred[act]) / np.maximum(pred[act], 1e-12)
                    errs[key].extend(np.abs(r).tolist()); exceed[key].extend((r > 0.20).tolist())
        res["aggregation"][str(win)] = {k: {"median_abs_rel_error": float(np.median(errs[k])), "p95_abs_rel_error": float(np.quantile(errs[k], .95)),
                                            "share_exceeding_prediction_by_20pct": float(np.mean(exceed[k]))} for k in errs}
        res["aggregation"][str(win)]["n_sources"] = len(names)
    json.dump(res, open(OUT / "traffic_escalation.json", "w"), indent=1)
    for win in ("60", "300"):
        print("window", win, "s")
        for n, r in res["sources"][win].items():
            if "spearman_median" in r:
                print(f"   {n:55s} windows {r['windows']:4d} rho {r['spearman_median']:+.2f} [{r['spearman_range'][0]:+.2f},{r['spearman_range'][1]:+.2f}] "
                      f"indep.err {r['independence_error_median']} q med {r['q_median']:.3f} p95 {r['q_p95']:.3f}")
        print("   aggregation", json.dumps(res["aggregation"][win]))
    print("saved")


if __name__ == "__main__":
    main()
