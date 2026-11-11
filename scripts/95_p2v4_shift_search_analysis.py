"""P2 v4, X13 analysis: search methods under escalation shift (scripts/94), same statistics as scripts/83 and 96.

Per instance (topology, rho, draw, scenario at its tight k): reference F_ref = the best objective of any feasible run in
X13 or the X4 placement (modelA_v4.jsonl). Per run: ratio F / F_ref if feasible, 2 if infeasible. Per instance and method:
mean ratio and feasible share over the seeds. Tests: Friedman over instances, Wilcoxon signed-rank of EHO-ACO against each
method with Holm correction, at the full budget and after 1,000 / 2,000 / 5,000 evaluations (anytime values use F only,
feasibility of intermediate placements is not recorded, so anytime ratios are F / F_ref without the infeasible rule).
Performance profiles (share of instances where a method is best, tau = 1.00, and within 5%, 25%).
Usage: 95_p2v4_shift_search_analysis.py [--partial]   Output: outputs_v5/p2/shift_search_stats.json."""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

D = "D:/shield_run/outputs_v5/p2"
R = [json.loads(l) for l in open(f"{D}/shift_search_runs.jsonl")]
X4 = [json.loads(l) for l in open(f"{D}/modelA_v4.jsonl")]
NAMES = {"sa": "SA", "hybrid_eho_aco": "EHO-ACO", "gwo_sa": "GWO+SA", "ga": "GA", "pso": "PSO"}
REF = "hybrid_eho_aco"
key = lambda r: (r["topology"], r["rho"], r["seed"], r["scen"])
fref = defaultdict(lambda: np.inf)
for r in R:
    if r["feasible"]:
        fref[key(r)] = min(fref[key(r)], r["F"])
present = {key(r) for r in R}
for x in X4:
    for sc, p in x["place"].items():
        k = (x["topology"], x["rho"], x["seed"], sc)
        if k in present and p and x["k_env"].get(sc) == p["k"]:      # X4's feasible placement at the same tight k
            fref[k] = min(fref[k], p["F"])
runs = defaultdict(lambda: defaultdict(list)); anyt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in R:
    i = key(r)
    if not np.isfinite(fref[i]):
        continue
    runs[i][r["alg"]].append(r["F"] / fref[i] if r["feasible"] else 2.0)
    for b, v in (r["anytime"] or {}).items():
        if v is not None:
            anyt[b][i][r["alg"]].append(v / fref[i])
meths = list(NAMES)
complete = [i for i in runs if all(len(runs[i][m]) >= (1 if "--partial" in sys.argv else 30) for m in meths)]
out = {"n_instances": len(complete), "n_runs": len(R), "full": {}, "anytime": {}}


def stats(table):
    M = np.array([[np.mean(table[i][m]) for m in meths] for i in complete])
    res = {"mean_ratio": {NAMES[m]: float(M[:, j].mean()) for j, m in enumerate(meths)}}
    best = M.min(1, keepdims=True)
    res["profile"] = {NAMES[m]: {t: float(np.mean(M[:, j] <= best[:, 0] * float(t) + 1e-12)) for t in ("1.00", "1.05", "1.25")}
                      for j, m in enumerate(meths)}
    if len(complete) >= 5:
        res["friedman_p"] = float(friedmanchisquare(*M.T).pvalue)
        pw = []
        for j, m in enumerate(meths):
            if m == REF:
                continue
            a, b = M[:, meths.index(REF)], M[:, j]; d = a - b
            p = float(wilcoxon(a, b).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
            pw.append([NAMES[m], int((d < -1e-12).sum()), int((d > 1e-12).sum()), p])
        order = np.argsort([x[3] for x in pw]); adj = [0.0] * len(pw); run = 0.0
        for kk, idx in enumerate(order):
            run = max(run, min(1.0, (len(pw) - kk) * pw[idx][3])); adj[idx] = run
        res["eho_vs"] = {x[0]: {"better": x[1], "worse": x[2], "p": x[3], "p_holm": adj[n]} for n, x in enumerate(pw)}
    return res


out["full"] = stats(runs)
out["full"]["feasible_share"] = {NAMES[m]: float(np.mean([np.mean([v < 2 for v in runs[i][m]]) for i in complete])) for m in meths}
for b in ("1000", "2000", "5000"):
    if all(all(len(anyt[b][i][m]) for m in meths) for i in complete):
        out["anytime"][b] = stats(anyt[b])
json.dump(out, open(f"{D}/shift_search_stats.json", "w"), indent=1)
print("instances", len(complete), "runs", len(R))
for part in ["full"] + list(out["anytime"]):
    s = out["full"] if part == "full" else out["anytime"][part]
    print(part, "mean ratio", {k: round(v, 3) for k, v in s["mean_ratio"].items()})
    print("   best share", {k: v["1.00"] for k, v in s["profile"].items()}, "friedman p", s.get("friedman_p"))
    for k, v in s.get("eho_vs", {}).items():
        print("   EHO-ACO vs", k, v)
if "feasible_share" in out["full"]:
    print("feasible share", {k: round(v, 3) for k, v in out["full"]["feasible_share"].items()})
