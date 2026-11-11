"""P2 v4, X8 anytime analysis (replaces an unsaved ad hoc test): quality over time on the 12 large instances.

Per run, the ratio z / z_ref after 1k, 2k, 5k, 10k and 20k evaluations (z_ref as scripts/83; an infeasible incumbent
scores 2). Per instance and method: mean over the 30 seeds. Reported per budget: the mean over instances of these
per-instance means, and for EHO-ACO (Algorithm 1 setting) against every other method the number of instances where
it is better / worse, a Wilcoxon signed-rank test over instances, and Holm-corrected p-values across the other methods
at that budget. Output: outputs_v5/p2/search_anytime.json."""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

D = "D:/shield_run/outputs_v5/p2"
R = [json.loads(l) for l in open(f"{D}/search_runs.jsonl")]
try:      # PSO-SA over-time records recovered by scripts/97 (same runs, history kept)
    _ps = {(x["topology"], x["K"], x["p"], x["seed"]): x["anytime"] for x in map(json.loads, open(f"{D}/psosa_anytime_runs.jsonl"))}
    for _r in R:
        if _r["alg"] == "pso_sa" and (_r["topology"], _r["K"], _r["p"], _r["seed"]) in _ps:
            _r["anytime"] = _ps[(_r["topology"], _r["K"], _r["p"], _r["seed"])]
except FileNotFoundError:
    pass
opt = json.load(open("D:/shield_run/outputs_v3/killi_rao/optimum.json"))
old = []
for f in ("D:/shield_run/outputs_v3/killi_rao/large_runs.jsonl", "D:/shield_run/outputs_v3/placement_cv/runs.jsonl"):
    try:
        old += [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        pass
inst = sorted({(r["topology"], r["K"], r["p"]) for r in R})
small = {i for i in inst if f"{i[0]}|{i[1]}|{i[2]}" in opt and opt[f"{i[0]}|{i[1]}|{i[2]}"]["status"] == 0}
large = [i for i in inst if i not in small and i[0] in ("Cogentco", "Kdl")]
best = defaultdict(lambda: np.inf)
for r in R + old:
    z = r.get("z", r.get("best_F", np.inf))
    if z is not None and z < 1000:
        best[(r["topology"], r.get("K", 2), r["p"])] = min(best[(r["topology"], r.get("K", 2), r["p"])], z)
per = defaultdict(lambda: defaultdict(list))            # (method, budget) -> instance -> ratios
for r in R:
    i = (r["topology"], r["K"], r["p"])
    if i not in large or not r.get("anytime"):
        continue
    for b, z in r["anytime"].items():
        if z is not None:
            per[(f"{r['alg']}:{r['cand']}", int(b))][i].append(2.0 if z >= 1000 else z / best[i])
meth = sorted({m for m, _ in per}); budgets = sorted({b for _, b in per}); ref = "hybrid_eho_aco:tuned"
out = {"n_large": len(large), "budgets": budgets, "mean": {}, "tests": {}}
for b in budgets:
    M = {m: np.array([np.mean(per[(m, b)][i]) for i in large]) for m in meth if all(per[(m, b)][i] for i in large)}
    out["mean"][b] = {m: float(v.mean()) for m, v in M.items()}
    pw = []
    for m in M:
        if m == ref:
            continue
        d = M[ref] - M[m]
        p = float(wilcoxon(M[ref], M[m]).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        pw.append([m, int((d < -1e-12).sum()), int((d > 1e-12).sum()), p])
    order = np.argsort([x[3] for x in pw]); adj = [0.0] * len(pw); run = 0.0
    for k, idx in enumerate(order):
        run = max(run, min(1.0, (len(pw) - k) * pw[idx][3])); adj[idx] = run
    out["tests"][b] = {x[0]: {"better": x[1], "worse": x[2], "p": x[3], "p_holm": adj[n]} for n, x in enumerate(pw)}
    print(b, {m: round(v, 3) for m, v in out["mean"][b].items()})
    for m, t in out["tests"][b].items():
        print("   EHO-ACO(Alg.1) vs", m, t)
json.dump(out, open(f"{D}/search_anytime.json", "w"), indent=1)
print("saved")
