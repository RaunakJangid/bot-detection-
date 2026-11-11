"""P2 v4: X13 under-shift benchmark (scripts/94, shift_search_runs.jsonl), objective quality reported separately from
feasibility (review request). Per algorithm, over all instance-runs:
  feasible yield = share of runs whose final placement is feasible (overall and per scenario);
  conditional quality = F / F_ref on feasible runs only (F_ref = best feasible F of any run of any algorithm on the instance);
  paired comparison on instances where both algorithms have >= 1 feasible run: per-instance mean conditional ratio,
  Wilcoxon EHO-ACO vs each other algorithm with Holm correction (wins/losses counted, ties within 1e-12);
  per-instance feasible-yield difference EHO-ACO - other, Wilcoxon + Holm.
Output: outputs_v5/p2/shift_feasibility.json."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("D:/shield_run/outputs_v5/p2")


def holm(ps):
    order = np.argsort(ps); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    R = [json.loads(l) for l in open(OUT / "shift_search_runs.jsonl")]
    key = lambda r: (r["topology"], r["rho"], r["seed"], r["scen"])
    algs = sorted({r["alg"] for r in R}); ref = defaultdict(lambda: np.inf)
    for r in R:
        if r["feasible"]:
            ref[key(r)] = min(ref[key(r)], r["F"])
    feas = defaultdict(list); cond = defaultdict(list)
    for r in R:
        feas[(key(r), r["alg"])].append(bool(r["feasible"]))
        if r["feasible"]:
            cond[(key(r), r["alg"])].append(r["F"] / ref[key(r)])
    inst = sorted({key(r) for r in R}); out = {"n_instances": len(inst), "n_runs": len(R), "algorithms": algs, "per_alg": {}}
    for a in algs:
        fy = [np.mean(feas[(i, a)]) for i in inst if feas[(i, a)]]
        cr = [x for i in inst for x in cond[(i, a)]]
        by_scen = defaultdict(list)
        for i in inst:
            by_scen[i[3]] += feas[(i, a)]
        out["per_alg"][a] = {"feasible_yield": float(np.mean([x for i in inst for x in feas[(i, a)]])),
                             "instances_with_any_feasible": int(sum(any(feas[(i, a)]) for i in inst)),
                             "conditional_ratio_mean": float(np.mean(cr)) if cr else None,
                             "conditional_ratio_median": float(np.median(cr)) if cr else None,
                             "feasible_yield_by_scenario": {s: float(np.mean(v)) for s, v in sorted(by_scen.items())}}
    E = "hybrid_eho_aco" if "hybrid_eho_aco" in algs else [a for a in algs if "eho" in a.lower()][0]
    pq, pf = [], []
    for a in algs:
        if a == E:
            continue
        both = [i for i in inst if cond[(i, E)] and cond[(i, a)]]
        x = np.array([np.mean(cond[(i, E)]) for i in both]); y = np.array([np.mean(cond[(i, a)]) for i in both]); d = x - y
        p = float(wilcoxon(x, y).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        pq.append((a, len(both), int((d < -1e-12).sum()), int((d > 1e-12).sum()), p, float(np.median(d))))
        fx = np.array([np.mean(feas[(i, E)]) for i in inst]); fz = np.array([np.mean(feas[(i, a)]) for i in inst]); df = fx - fz
        p2 = float(wilcoxon(fx, fz).pvalue) if np.any(np.abs(df) > 1e-12) else 1.0
        pf.append((a, int((df > 1e-12).sum()), int((df < -1e-12).sum()), p2, float(np.mean(df))))
    hq = holm([t[4] for t in pq]); hf = holm([t[3] for t in pf])
    out["reference_alg"] = E
    out["conditional_quality_vs"] = {t[0]: {"instances_both_feasible": t[1], "eho_better": t[2], "eho_worse": t[3], "p": t[4],
                                            "p_holm": hq[n], "median_ratio_diff": t[5]} for n, t in enumerate(pq)}
    out["feasible_yield_vs"] = {t[0]: {"eho_higher": t[1], "eho_lower": t[2], "p": t[3], "p_holm": hf[n], "mean_yield_diff": t[4]}
                                for n, t in enumerate(pf)}
    json.dump(out, open(OUT / "shift_feasibility.json", "w"), indent=1)
    for a, v in out["per_alg"].items():
        print(f"{a:16s} yield {v['feasible_yield']:.3f} inst>=1 feasible {v['instances_with_any_feasible']}/{len(inst)} "
              f"cond ratio mean {v['conditional_ratio_mean']:.4f} median {v['conditional_ratio_median']:.4f}")
    for a, v in out["conditional_quality_vs"].items():
        print("  conditional quality", E, "vs", a, v)
    for a, v in out["feasible_yield_vs"].items():
        print("  feasible yield", E, "vs", a, v)


if __name__ == "__main__":
    main()
