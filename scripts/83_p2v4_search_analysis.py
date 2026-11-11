"""P2 v4, X8 analysis: search methods on the published FFCCP model, 30 seeds (outputs_v5/p2/search_runs.jsonl).
Reference z_ref: the proven optimum (outputs_v3/killi_rao/optimum.json) on the small instances, the best feasible value
found by any run (this experiment, outputs_v3/killi_rao/large_runs.jsonl, outputs_v3/placement_cv/runs.jsonl) on the
large ones. Per run: ratio r = z / z_ref, r = 2 for an infeasible run (z >= 1000, the penalty range). Per instance and
method: mean r over the 30 seeds. Tests: Friedman over instances on the mean ratios (all 55, and the 12 large), average
ranks, and Wilcoxon signed-rank tests of every pair against EHO-ACO (cross-validated setting) with Holm correction.
Anytime: mean ratio after 1k to 20k evaluations on the large instances. Output: outputs_v5/p2/search_stats.json."""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

R = [json.loads(l) for l in open("D:/shield_run/outputs_v5/p2/search_runs.jsonl")]
opt = json.load(open("D:/shield_run/outputs_v3/killi_rao/optimum.json"))
old = []
for f in ("D:/shield_run/outputs_v3/killi_rao/large_runs.jsonl", "D:/shield_run/outputs_v3/placement_cv/runs.jsonl"):
    try:
        old += [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        pass
inst = sorted({(r["topology"], r["K"], r["p"]) for r in R})
small = [i for i in inst if f"{i[0]}|{i[1]}|{i[2]}" in opt and opt[f"{i[0]}|{i[1]}|{i[2]}"]["status"] == 0]
large = [i for i in inst if i not in small]
best = defaultdict(lambda: np.inf)
for r in R + old:
    key = (r["topology"], r.get("K", 2), r["p"])
    z = r.get("z", r.get("best_F", np.inf))
    if z is not None and z < 1000:
        best[key] = min(best[key], z)
zref = {i: (opt[f"{i[0]}|{i[1]}|{i[2]}"]["z"] if i in small else best[i]) for i in inst}
meth = sorted({(r["alg"], r["cand"]) for r in R})
name = {m: f"{m[0]}:{m[1]}" for m in meth}
ratio = defaultdict(list); hit = defaultdict(list); feas = defaultdict(list); anyt = defaultdict(lambda: defaultdict(list))
for r in R:
    i, m = (r["topology"], r["K"], r["p"]), (r["alg"], r["cand"])
    rr = 2.0 if r["z"] >= 1000 else r["z"] / zref[i]
    ratio[(i, m)].append(rr)
    if i in small:
        hit[m].append(abs(r["z"] - zref[i]) <= 1e-6 * max(1.0, zref[i]))
    else:
        feas[m].append(r["z"] < 1000)
        if r["anytime"]:
            for c, z in r["anytime"].items():
                if z is not None:
                    anyt[m][c].append(2.0 if z >= 1000 else z / zref[i])
out = {"n_small": len(small), "n_large": len(large), "methods": {}, "tests": {}}
for m in meth:
    rs = [np.mean(ratio[(i, m)]) for i in inst]
    rl = [np.mean(ratio[(i, m)]) for i in large]
    out["methods"][name[m]] = {"mean_ratio_all": float(np.mean(rs)), "mean_ratio_large": float(np.mean(rl)),
                               "small_opt_hit_pct": 100 * float(np.mean(hit[m])),
                               "large_feasible_pct": 100 * float(np.mean(feas[m])),
                               "large_gap_pct_feasible": float(100 * np.mean([x - 1 for i in large for x in ratio[(i, m)] if x < 2])),
                               "anytime_large": {c: float(np.mean(v)) for c, v in sorted(anyt[m].items(), key=lambda kv: int(kv[0]))}}
ref_m = ("hybrid_eho_aco", "ls_every5")
for label, subset in (("all", inst), ("large", large), ("small", small)):
    M = np.array([[np.mean(ratio[(i, m)]) for m in meth] for i in subset])
    stat, p = friedmanchisquare(*M.T)
    ranks = np.mean([rankdata(row) for row in M], axis=0)
    pw = []
    for j, m in enumerate(meth):
        if m == ref_m:
            continue
        a, b = M[:, meth.index(ref_m)], M[:, j]
        d = a - b
        p_ = float(wilcoxon(a, b).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        pw.append([name[m], int((d < -1e-12).sum()), int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), p_])
    order = np.argsort([x[4] for x in pw]); k = len(pw)
    adj = [0.0] * k; run = 0.0
    for rank_i, idx in enumerate(order):                       # Holm step-down
        run = max(run, min(1.0, (k - rank_i) * pw[idx][4])); adj[idx] = run
    out["tests"][label] = {"n_instances": len(subset), "friedman_chi2": float(stat), "friedman_p": float(p),
                           "avg_rank": {name[m]: float(r) for m, r in zip(meth, ranks)},
                           "eho_aco_cv_vs": {x[0]: {"eho_better": x[1], "eho_worse": x[2], "ties": x[3], "p": x[4],
                                                    "p_holm": adj[n]} for n, x in enumerate(pw)}}
json.dump(out, open("D:/shield_run/outputs_v5/p2/search_stats.json", "w"), indent=1)
print(f"{len(small)} small, {len(large)} large instances")
for m, v in out["methods"].items():
    print(f"{m:26s} ratio all {v['mean_ratio_all']:.4f} large {v['mean_ratio_large']:.4f} | opt hit {v['small_opt_hit_pct']:.1f}% | large feasible {v['large_feasible_pct']:.1f}% gap {v['large_gap_pct_feasible']:.2f}% | anytime {[round(x, 3) for x in v['anytime_large'].values()]}")
for label, t in out["tests"].items():
    print(label, "Friedman p", f"{t['friedman_p']:.2e}", "ranks", {k: round(v, 2) for k, v in t["avg_rank"].items()})
    for k, v in t["eho_aco_cv_vs"].items():
        print("   EHO-ACO(cv) vs", k, v)
