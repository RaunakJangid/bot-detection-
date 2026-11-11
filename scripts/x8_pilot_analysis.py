"""Cycle 1 pilot analysis (K4'): computes P1 to P6 exactly as frozen in papers/novelty/RESEARCH_LOG.md (Stage 1).

Live flows = flows from the 6 benign hosts; control flows = flows from 10.0.0.90. Commissioning = flows before the switch.
Commissioned threshold = 99th percentile of commissioning live scores. Windows after the switch: [30, 90), [90, 150),
[150, 210) s. Per window:
  baseline a  live KS: two-sample KS distance of window live scores vs commissioning live scores / alpha = 0.001 critical
  baseline b  live PSI: mean over 6 numeric Zeek fields (log1p, commissioning deciles) of the population stability index
  baseline c  live conformal rejection rate: share of window live flows with conformal p-value < 0.01 against the
              commissioning live scores
  ours        control KS: KS distance of window control scores vs commissioning control scores / its critical value
  (A1 ablation = baseline a: the same KS test on live flows, every other component unchanged)
Attribution AUROC: S windows (S1, S2 runs) labelled 1 against T windows (T1 runs) labelled 0, pooled and per repeat.
Output: papers/novelty/runs/cycle_1/pilot_summary.json."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

D = "D:/shield_run/papers/novelty/runs/cycle_1"
BENIGN = {f"10.0.0.{i}" for i in range(1, 7)}
CTRL = "10.0.0.90"
NUMF = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "resp_pkts"]
WIN = [(30, 90), (90, 150), (150, 210)]


def ks(a, b):
    a, b = np.sort(a), np.sort(b); v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def crit(n1, n2):
    return 1.949 * np.sqrt((n1 + n2) / (n1 * n2))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def psi(ref, cur):
    vals = []
    for k in range(ref.shape[1]):
        r, c = np.log1p(ref[:, k]), np.log1p(cur[:, k])
        edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        pr = np.histogram(np.clip(r, edges[0], edges[-1]), edges)[0] / len(r) + 1e-4
        pc = np.histogram(np.clip(c, edges[0], edges[-1]), edges)[0] / len(c) + 1e-4
        vals.append(float(np.sum((pc - pr) * np.log(pc / pr))))
    return float(np.mean(vals)) if vals else 0.0


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def analyse(path):
    d = json.load(open(path)); F = d["flows"]; ts = d["t_switch"]
    live = [f for f in F if f["src"] in BENIGN]; ctrl = [f for f in F if f["src"] == CTRL]
    wl = np.array([f["score"] for f in live if f["t"] < ts]); wc = np.array([f["score"] for f in ctrl if f["t"] < ts])
    wX = np.array([[num(f[k]) for k in NUMF] for f in live if f["t"] < ts])
    thr = float(np.quantile(wl, 0.99))
    post = [f for f in live if f["t"] >= ts + 30]
    fpr = float(np.mean([f["score"] > thr for f in post])) if post else None
    # P1 KS warning (P2 reproduction): every 50 post-window live flows, last 300 vs commissioning
    def warn_share(stream):
        w0 = np.array([f["score"] for f in stream if f["t"] < ts]); c_, rec, k = [], [], 0
        for f in (f for f in stream if f["t"] >= ts):
            rec.append(f["score"]); rec = rec[-300:]; k += 1
            if k % 50 == 0 and len(rec) >= 100:
                c_.append(ks(np.array(rec), w0) > crit(len(rec), len(w0)))
        return c_
    checks = warn_share([f for f in F if f["src"] != CTRL])            # P1 definition: every non-server flow
    checks_v4 = warn_share(live)                                       # benign IPv4 hosts only
    wins = []
    for a, b in WIN:
        L = [f for f in live if ts + a <= f["t"] < ts + b]; C = [f for f in ctrl if ts + a <= f["t"] < ts + b]
        ls = np.array([f["score"] for f in L]); cs = np.array([f["score"] for f in C])
        X = np.array([[num(f[k]) for k in NUMF] for f in L])
        pv = np.array([(1 + np.sum(wl >= s)) / (len(wl) + 1) for s in ls])
        wins.append({"n_live": len(L), "n_ctrl": len(C),
                     "live_ks": ks(ls, wl) / crit(len(ls), len(wl)) if len(ls) and len(wl) else None,
                     "live_psi": psi(wX, X) if len(X) else None,
                     "live_rej": float(np.mean(pv < 0.01)) if len(pv) else None,
                     "ctrl_ks": ks(cs, wc) / crit(len(cs), len(wc)) if len(cs) and len(wc) else None})
    sent = [l.split()[1] for l in d["ctrl_sent"]]
    expected = sum(s in ("http", "udp", "icmp") for s in sent) + 5 * sent.count("syn5")
    c0, c1 = d["srv_counters_switch"], d["srv_counters_end"]
    return {"cond": d["cond"], "rep": d["rep"], "n_flows": d["n_flows"], "n_live": len(live), "n_ctrl": len(ctrl),
            "ctrl_expected": expected, "ctrl_seen_share": len(ctrl) / expected if expected else None,
            "warm_max_t": max(f["t"] for f in F if f["t"] < ts), "t_switch": ts,
            "srv_post_pkts": (c1[0] - c0[0]) + (c1[1] - c0[1]), "thr": thr, "fpr": fpr,
            "ks_warning_share": float(np.mean(checks)) if checks else None, "n_checks": len(checks),
            "ks_warning_share_ipv4": float(np.mean(checks_v4)) if checks_v4 else None,
            "n_ipv6_flows": sum(f["src"].startswith("fe80") for f in F), "windows": wins,
            "mtime": os.path.getmtime(path)}


def main():
    R = [analyse(p) for p in sorted(glob.glob(f"{D}/*_r[123].json"))]
    by = defaultdict(list)
    for r in R:
        by[r["cond"]].append(r)
    out = {"runs": R}
    # P1 data sanity
    r0p = np.mean([r["srv_post_pkts"] for r in by["R0"]])
    out["P1"] = {"n_runs": len(R), "min_live": min(r["n_live"] for r in R),
                 "min_ctrl_seen_share": min(r["ctrl_seen_share"] for r in R),
                 "srv_pkts_vs_R0": {c: [round(r["srv_post_pkts"] / r0p - 1, 3) for r in by[c]] for c in ("S1", "S2")},
                 "no_leakage": all(r["warm_max_t"] < r["t_switch"] for r in R)}
    out["P1"]["pass"] = (len(R) == 12 and out["P1"]["min_live"] >= 1000 and out["P1"]["min_ctrl_seen_share"] >= 0.95
                         and all(abs(x) <= 0.05 for v in out["P1"]["srv_pkts_vs_R0"].values() for x in v)
                         and out["P1"]["no_leakage"])
    # P2 baseline reproduction
    ksw = [r["ks_warning_share"] for r in by["R0"]]
    pooled = float(np.sum([r["ks_warning_share"] * r["n_checks"] for r in by["R0"]]) / np.sum([r["n_checks"] for r in by["R0"]]))
    out["P2"] = {"R0_ks_warning_share_pooled": pooled, "per_run": ksw, "pass": 0.745 <= pooled <= 0.845,
                 "R0_ipv4_only_per_run": [r["ks_warning_share_ipv4"] for r in by["R0"]],
                 "R0_ipv6_flows": [r["n_ipv6_flows"] for r in by["R0"]]}
    # P3 gap: (a) FPR, (b) baselines' attribution AUROC
    f = {c: [r["fpr"] for r in by[c]] for c in by}
    a_ok = {c: (np.mean(f[c]) >= np.mean(f["R0"]) + 0.01 and min(f[c]) > max(f["R0"])) for c in ("S1", "S2")}
    S = [w for c in ("S1", "S2") for r in by[c] for w in r["windows"]]; Tw = [w for r in by["T1"] for w in r["windows"]]
    au = {m: auroc([w[m] for w in S], [w[m] for w in Tw]) for m in ("live_ks", "live_psi", "live_rej", "ctrl_ks")}
    per_rep = {m: [auroc([w[m] for c in ("S1", "S2") for r in by[c] if r["rep"] == k for w in r["windows"]],
                         [w[m] for r in by["T1"] if r["rep"] == k for w in r["windows"]]) for k in (1, 2, 3)]
               for m in au}
    best_b = max(("live_ks", "live_psi", "live_rej"), key=lambda m: au[m])
    out["P3"] = {"fpr": f, "fpr_a_pass": a_ok, "auroc": au, "auroc_per_rep": per_rep, "best_baseline": best_b,
                 "pass": any(a_ok.values()) and au[best_b] < 0.80}
    out["P4"] = {"ours": au["ctrl_ks"], "best_baseline": au[best_b],
                 "pass": au["ctrl_ks"] >= 0.90 and au["ctrl_ks"] - au[best_b] >= 0.20
                 and min(per_rep["ctrl_ks"]) > max(per_rep[best_b])}
    out["P5"] = {"ours": au["ctrl_ks"], "A1_live_ks": au["live_ks"], "drop": au["ctrl_ks"] - au["live_ks"],
                 "pass": au["ctrl_ks"] - au["live_ks"] >= 0.20}
    ts_ = sorted(r["mtime"] for r in R)
    per_run = float(np.median(np.diff(ts_))) if len(ts_) > 1 else None
    out["P6"] = {"seconds_per_run": per_run, "projected_full_h": 110 * per_run / 3600 + 1 if per_run else None}
    out["P6"]["pass"] = out["P6"]["projected_full_h"] is not None and out["P6"]["projected_full_h"] <= 120 - 12
    out["sanity_R0_windows"] = {m: [w[m] for r in by["R0"] for w in r["windows"]] for m in ("live_ks", "ctrl_ks")}
    json.dump(out, open(f"{D}/pilot_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        print(k, json.dumps(out[k], default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("R0 windows", json.dumps(out["sanity_R0_windows"], default=float))


if __name__ == "__main__":
    main()
