"""P2 v4, presentation of X8: Dolan-More performance profiles of the search methods on the FFCCP benchmark.

For each instance and method: the mean ratio z / z_ref over the 30 seeds (infeasible = 2, as scripts/83), at the full
budget (20,000 evaluations, all methods) and after 5,000 evaluations (methods with history; PSO-SA has none). The
performance ratio of a method on an instance is its mean ratio divided by the best mean ratio of any method on that
instance; the profile rho_m(tau) is the share of instances with performance ratio <= tau. Reported for all 55 instances
and for the 12 large ones, tau in {1.00, 1.01, 1.02, 1.05, 1.10, 1.25}.
Output: outputs_v5/p2/perf_profiles.json and figure papers/src/fig_p2v4_profiles.png."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
key = lambda i: f"{i[0]}|{i[1]}|{i[2]}"
small = {i for i in inst if key(i) in opt and opt[key(i)]["status"] == 0}
large = [i for i in inst if i not in small and i[0] in ("Cogentco", "Kdl")]
best = defaultdict(lambda: np.inf)
for r in R + old:
    z = r.get("z", r.get("best_F", np.inf))
    if z is not None and z < 1000:
        best[(r["topology"], r.get("K", 2), r["p"])] = min(best[(r["topology"], r.get("K", 2), r["p"])], z)
zref = {i: (opt[key(i)]["z"] if i in small else best[i]) for i in inst}
NAMES = {"sa:cool999": "SA", "hybrid_eho_aco:tuned": "EHO-ACO (Alg. 1)", "gwo_sa:tuned": "GWO+SA",
         "hybrid_eho_aco:ls_every5": "EHO-ACO (CV)", "pso_sa:default": "PSO-SA", "ga:tuned": "GA"}
TAUS = (1.00, 1.01, 1.02, 1.05, 1.10, 1.25)
COL = {"sa:cool999": "#1f77b4", "hybrid_eho_aco:tuned": "#ff7f0e", "gwo_sa:tuned": "#2ca02c",
       "hybrid_eho_aco:ls_every5": "#d62728", "pso_sa:default": "#9467bd", "ga:tuned": "#8c564b"}


def mean_ratio(budget):
    acc = defaultdict(list)
    for r in R:
        i, m = (r["topology"], r["K"], r["p"]), f"{r['alg']}:{r['cand']}"
        if budget is None:
            z = r["z"]
        else:
            if not r.get("anytime") or r["anytime"].get(str(budget)) is None:
                continue
            z = r["anytime"][str(budget)]
        acc[(i, m)].append(2.0 if z >= 1000 else z / zref[i])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def profiles(mr, subset):
    meths = sorted({m for (_, m) in mr if all((i, m) in mr for i in subset)})
    out = {}
    for m in meths:
        pr = [mr[(i, m)] / min(mr[(i, mm)] for mm in meths) for i in subset]
        out[m] = {str(t): float(np.mean([x <= t + 1e-12 for x in pr])) for t in TAUS}
    return out


res = {}
for label, b in (("full_20k", None), ("budget_5k", 5000)):
    mr = mean_ratio(b)
    res[label] = {"all": profiles(mr, inst), "large": profiles(mr, large)}
json.dump(res, open(f"{D}/perf_profiles.json", "w"), indent=1)
for label in res:
    for sub in ("all", "large"):
        print(label, sub)
        for m, v in res[label][sub].items():
            print(f"   {NAMES.get(m, m):18s}", "  ".join(f"t={t}:{v[str(t)]:.2f}" for t in TAUS))

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6), dpi=300)
for ax, (label, title) in zip(axes, (("budget_5k", "(a) large, 5,000 evaluations*"), ("full_20k", "(b) large, 20,000 evaluations"))):
    mr = mean_ratio(None if label == "full_20k" else 5000)
    meths = [m for m in NAMES if all((i, m) in mr for i in large)]
    taus = np.linspace(1.0, 1.6, 121)
    for m in meths:
        pr = np.array([mr[(i, m)] / min(mr[(i, mm)] for mm in meths) for i in large])
        ax.step(taus, [(pr <= t + 1e-12).mean() for t in taus], where="post", lw=0.9, label=NAMES[m], color=COL[m])
    ax.set_title(title, fontsize=6); ax.set_xlabel(r"$\tau$ (ratio to best method)", fontsize=6)
    ax.set_ylim(0, 1.02); ax.tick_params(labelsize=5.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Share of instances", fontsize=6); axes[1].legend(fontsize=4.2, frameon=False, loc="lower right")
fig.tight_layout(pad=0.2); fig.savefig("D:/shield_run/papers/src/fig_p2v4_profiles.png")
print("saved")
