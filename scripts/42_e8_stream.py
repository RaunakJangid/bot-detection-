"""E8: can a detector hold a prespecified false-alarm budget on TIME-ORDERED traffic from new captures?
Controlled comparison of operating-point maintenance strategies (fixed before the run).

Data: per-flow scores saved by 41_iot23_shift.py (IoT-23; real Zeek timestamps), designs `same_family` (new captures,
known malware families: benign drift) and `capture_shift` (new captures and unseen families). Each test capture is
replayed in timestamp order, prequentially (decide first, then update). Target FPR alpha = 1%.
Methods (each applied on top of every saved model's attack score):
  fixed        threshold from validation benign flows (train captures)
  burnin_rand  conformal quantile of K benign flows sampled at random from the capture (exchangeable; optimistic)
  burnin_time  conformal quantile of the FIRST K benign flows in time (clean commissioning period)
  roll_free    label-free rolling: every B flows, threshold = contamination-corrected quantile of the last W scores
               (pi_hat = share above the validation 0.1%-FPR threshold; q = pi_hat + alpha (1 - pi_hat))
  roll_audit   delayed-label rolling: an analyst audits a random 5% of flows, labels arrive after D seconds; threshold
               = conformal quantile of the last W_a audited benign flows (Jin & Lee 2026 propose this as future work)
  nguyen       re-implementation of Nguyen et al. 2026 (Discover AI 6:1394) feedback-guided decision-boundary
               adaptation: EMA z-scores on d = 4 features (chosen by validation univariate AUC; the paper does not name
               them), feature thresholds gamma, global gate alpha, batch B = 200 updates from mean normalised entropy
               and a Hoeffding drift signal, all parameters from their Table 3. Samples passing the gate go to the
               model at its validation 1% threshold (their classifier uses argmax; we keep the budgeted threshold).
Metrics per model x capture: realised FPR, recall, FPR per hour (std; share of hours above 1.5 alpha), update cost.
Decision rule: a method "holds the budget" on a design if the benign-weighted pooled FPR over captures is <= 1.5% for
the median model AND median recall loss vs fixed <= 5 points."""

import math
import time

import numpy as np

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("e8")
ALPHA, K, B, W, WA, AUDIT, DELAY = 0.01, 1000, 200, 2000, 500, 0.05, 600.0
NG = dict(eta=0.01, B=200, rg=0.008, kg=0.025, ra=0.0016, ka=0.006, gmin=1.5, gmax=4.0, amin=0.05, amax=0.5,
          a0=0.16, g0=(2.2, 2.5, 2.8, 1.9), delta=0.05, n0=300, n1=100)


def conformal(s, a=ALPHA):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) or len(s) == 0 else np.sort(s)[k - 1]


def run_rolling_free(s, t_fix, t_hi):
    thr = np.full(len(s), t_fix)
    cur = t_fix
    for i in range(0, len(s), B):
        thr[i:i + B] = cur
        lo = max(0, i + B - W)
        win = s[lo:i + B]
        if len(win) >= W // 2:
            pi = float((win > t_hi).mean())
            cur = float(np.quantile(win, 1 - min(pi + ALPHA * (1 - pi), 0.999)))
    return s > thr


def run_rolling_audit(s, y, ts, t_fix, rng):
    audited = rng.random(len(s)) < AUDIT
    thr = np.empty(len(s)); cur = t_fix; pool = []
    j = 0  # pointer into audited flows whose label has arrived
    aud_idx = np.flatnonzero(audited)
    for i in range(0, len(s), B):
        thr[i:i + B] = cur
        now = ts[min(i + B, len(s)) - 1]
        while j < len(aud_idx) and aud_idx[j] < i + B and ts[aud_idx[j]] + DELAY <= now:
            if y[aud_idx[j]] == 0:
                pool.append(s[aud_idx[j]])
            j += 1
        if len(pool) >= 100:
            cur = float(conformal(np.asarray(pool[-WA:])))
    return s > thr


def run_nguyen(s, X4, mu0, sd0, t_fix):
    """Fixed classifier (budgeted threshold) behind an adaptive abnormality gate (Nguyen et al. 2026, Alg. 1)."""
    p = NG; d = X4.shape[1]
    Z = (X4 - mu0) / sd0                                  # standardised with train-capture statistics
    mu, var = np.zeros(d), np.ones(d)
    g = np.asarray(p["g0"], float); a = p["a0"]
    eps_b = math.sqrt(0.5 * math.log(2 / p["delta"]) * (1 / p["n0"] + 1 / p["n1"]))
    pred = np.zeros(len(s), bool); rhist = []; U = []
    pc = np.clip(s, 1e-12, 1 - 1e-12)
    ent = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)) / math.log(2)
    for t in range(len(s)):
        z = (Z[t] - mu) / (np.sqrt(var) + 1e-6)
        r = float((np.abs(z) >= g).mean())
        if r >= a:
            pred[t] = True; U.append(1.0)
        else:
            pred[t] = s[t] > t_fix; U.append(float(ent[t]))
        rhist.append(r)
        mu = (1 - p["eta"]) * mu + p["eta"] * Z[t]
        var = (1 - p["eta"]) * var + p["eta"] * (Z[t] - mu) ** 2
        if (t + 1) % p["B"] == 0:
            Ub = float(np.mean(U[-p["B"]:]))
            Q = 0.0
            if len(rhist) >= p["n0"] + p["n1"]:
                r1 = np.mean(rhist[-p["n1"]:]); r0 = np.mean(rhist[-(p["n0"] + p["n1"]):-p["n1"]])
                D = max(0.0, abs(r1 - r0) - eps_b); Q = float(np.clip(D / (0.25 - eps_b), 0, 1))
            g = np.clip(g + p["rg"] * Ub - p["kg"] * Q, p["gmin"], p["gmax"])
            a = float(np.clip(a + p["ra"] * Ub - p["ka"] * Q, p["amin"], p["amax"]))
    return pred


def hourly(pred, y, ts):
    hrs = np.floor((ts - ts.min()) / 3600).astype(int)
    f = []
    for h in np.unique(hrs):
        m = (hrs == h) & (y == 0)
        if m.sum() >= 50:
            f.append(pred[m].mean())
    f = np.asarray(f)
    return (float(f.std()) if len(f) > 1 else None, float((f > 1.5 * ALPHA).mean()) if len(f) else None, int(len(f)))


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--min-benign", type=int, default=1500)
    p.add_argument("--k", type=int, default=1000, help="burn-in size (sensitivity)")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    global K
    K = args.k
    P = Paths(args.smoke)
    out = {}
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=False)
        models, sv, st, yv, yt = list(Z["models"]), Z["sv"], Z["st"], Z["yv"], Z["yt"]
        cap, ts, Xv, Xt, feats = Z["capture"], Z["ts"], Z["Xv"], Z["Xt"], list(Z["features"])
        # Nguyen gate features: 4 most separating single features on VALIDATION (paper leaves them unspecified)
        from sklearn.metrics import roc_auc_score
        aucs = []
        for j in range(Xv.shape[1]):
            try:
                a_ = roc_auc_score(yv, Xv[:, j]); aucs.append(max(a_, 1 - a_))
            except ValueError:
                aucs.append(0.5)
        f4 = list(np.argsort(aucs)[::-1][:4])
        mu0, sd0 = Xv[yv == 0][:, f4].mean(0), Xv[yv == 0][:, f4].std(0) + 1e-6
        caps = [c for c in np.unique(cap) if ((cap == c) & (yt == 0)).sum() >= args.min_benign + K]
        log.info("%s: %d models; captures with enough benign: %s; gate features %s", design, len(models), caps,
                 [feats[j] for j in f4])
        res = {"captures": caps, "gate_features": [feats[j] for j in f4], "models": {}}
        for mi, m in enumerate(models):
            t_fix = float(np.quantile(sv[mi][yv == 0], 1 - ALPHA)); t_hi = float(np.quantile(sv[mi][yv == 0], 1 - ALPHA / 10))
            rm = {k: [] for k in ("fixed", "burnin_rand", "burnin_time", "roll_free", "roll_audit", "nguyen")}
            cost = {k: 0.0 for k in rm}; nflows = 0
            for c in caps:
                idx = np.flatnonzero(cap == c); idx = idx[np.argsort(ts[idx], kind="stable")]
                s, y, tt = st[mi][idx], yt[idx], ts[idx]
                rng = np.random.default_rng(mi * 1000 + caps.index(c))
                ben_pos = np.flatnonzero(y == 0)
                first_k = ben_pos[:K]; rand_k = rng.choice(ben_pos, K, replace=False)
                keep_t = np.ones(len(s), bool); keep_t[first_k] = False
                keep_r = np.ones(len(s), bool); keep_r[rand_k] = False
                preds = {}
                t0 = time.perf_counter(); preds["fixed"] = s > t_fix; cost["fixed"] += time.perf_counter() - t0
                t0 = time.perf_counter(); preds["burnin_rand"] = s > conformal(s[rand_k]); cost["burnin_rand"] += time.perf_counter() - t0
                t0 = time.perf_counter(); preds["burnin_time"] = s > conformal(s[first_k]); cost["burnin_time"] += time.perf_counter() - t0
                t0 = time.perf_counter(); preds["roll_free"] = run_rolling_free(s, t_fix, t_hi); cost["roll_free"] += time.perf_counter() - t0
                t0 = time.perf_counter(); preds["roll_audit"] = run_rolling_audit(s, y, tt, t_fix, rng); cost["roll_audit"] += time.perf_counter() - t0
                t0 = time.perf_counter(); preds["nguyen"] = run_nguyen(s, Xt[idx][:, f4], mu0, sd0, t_fix); cost["nguyen"] += time.perf_counter() - t0
                nflows += len(s)
                for k, pr in preds.items():
                    ev = keep_r if k == "burnin_rand" else keep_t     # never score the burn-in flows themselves
                    b, a = ev & (y == 0), ev & (y == 1)
                    sd_h, over_h, nh = hourly(pr[ev], y[ev], tt[ev])
                    rm[k].append({"capture": str(c), "n_benign": int(b.sum()), "fp": int(pr[b].sum()),
                                  "fpr": float(pr[b].mean()), "recall": float(pr[a].mean()) if a.any() else None,
                                  "hourly_fpr_sd": sd_h, "share_hours_over": over_h, "hours": nh})
            summ = {}
            for k, rows in rm.items():
                nb = sum(r["n_benign"] for r in rows); fp = sum(r["fp"] for r in rows)
                rec = [r["recall"] for r in rows if r["recall"] is not None]
                summ[k] = {"pooled_fpr": fp / max(nb, 1), "median_recall": float(np.median(rec)) if rec else None,
                           "per_capture": rows, "us_per_flow": 1e6 * cost[k] / max(nflows, 1)}
            res["models"][m] = summ
            log.info("%s %s: pooled FPR %s", design, m, {k: round(100 * v["pooled_fpr"], 2) for k, v in summ.items()})
        out[design] = res
        save_json(out, P.v3 / "iot23" / f"e8_stream{args.tag}.json")


if __name__ == "__main__":
    main()
