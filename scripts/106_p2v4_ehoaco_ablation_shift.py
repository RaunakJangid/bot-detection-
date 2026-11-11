"""P2 v4: EHO-ACO component ablation on the under-shift benchmark (the instances of X13, scripts/94: Model A at the tight
controller counts of the 16% / 27% / 13% scenarios, demand draws 7 and 107; 52 instances).

Variants and the shared Algorithm 1 setting as scripts/105 (hybrid_eho_aco, hybrid_no_ants = EHO+SA, hybrid_no_ls =
EHO+ACO, eho, aco, aco_sa = ACO+SA); 30 seeds x 20,000 evaluations; best objective after 250, 500, 1k, 2k, 5k, 10k
evaluations; final placement feasibility. Analysis (--analyse): per instance mean ratio F/F_ref (F_ref = best feasible
value of any run of X13 or this ablation; infeasible final placement = 2), Friedman, Wilcoxon of the full method vs each
variant with Holm correction, at the full budget and every checkpoint (checkpoint ratios use F only).
Output: outputs_v5/p2/ablation_shift_runs.jsonl, ablation_shift_stats.json."""
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m94, m105 = _l("94_p2v4_shift_search.py"), _l("105_p2v4_ehoaco_ablation.py")
VARIANTS, CHECK, SEEDS, BUDGET = m105.VARIANTS, (250, 500, 1000, 2000, 5000, 10000), 30, 20_000


def _init():
    """pool initializer defined here (fixed 2026-11-08): spawn workers cannot unpickle m94._init, whose module name
    94_p2v4_shift_search is not importable; each worker re-imports this file, which loads its own m94."""
    m94._init()


def _run(t):
    import numpy as np
    from shield.placement.coupling import feasible, topology_sla
    from shield.placement.objective import Instance, PlacementProblem, make_demand
    from shield.placement.runner import topology_and_latency, tuned_cfg, with_params
    topo_name, rho, seed, scen, k, r, alg, s = t
    W = m94._W; P, pcfg, ccfg, tuned, mu0 = W["P"], W["pcfg"], W["ccfg"], W["tuned"], W["mu0"]
    topo, D = topology_and_latency(topo_name, str(P.topologies))
    lam = make_demand(topo.n, 1.0, pcfg["demand_sigma"], seed)
    sla = topology_sla(D, lam, ccfg.get("sla", {}), ccfg["k_ref"], pcfg["objective"]["sla_ms"])
    cfg = with_params(tuned_cfg(pcfg, "hybrid_eho_aco", tuned), {"objective.sla_ms": sla}); o = cfg["objective"]
    inst = Instance(topo.name, D, make_demand(topo.n, rho * ccfg["k_ref"] * mu0, cfg["demand_sigma"], seed), r * mu0, k, topo.coords)
    pr = PlacementProblem(inst, o["weights"], cfg["controller"]["u_max"], o["overload_penalty"], o["sla_ms"], o["n_reference"], seed=seed)
    res = m105.run_variant(alg, pr, topo, cfg, s, BUDGET)
    hist = getattr(res, "history", None) or []
    any_ = {str(c): (float(min(f for e, f in hist if e <= c)) if any(e <= c for e, f in hist) else None) for c in CHECK}
    return {"topology": topo_name, "rho": rho, "seed": seed, "scen": scen, "k": k, "alg": alg, "run": s, "F": float(res.best_F),
            "feasible": bool(feasible(pr.describe(res.best), k, ccfg)), "anytime": any_}


def run(workers):
    rows = [json.loads(l) for l in open(OUT / "modelA_v4.jsonl")]
    tasks = [(x["topology"], x["rho"], x["seed"], sc, int(x["k_env"][sc]), float(x["ratios"][sc]), a, s)
             for x in rows if x["seed"] in (7, 107) for sc in m94.SCEN if x["k_env"].get(sc) is not None
             for a in VARIANTS for s in range(SEEDS)]
    out_f = OUT / "ablation_shift_runs.jsonl"; done = set()
    if out_f.exists():
        done = {(r["topology"], r["rho"], r["seed"], r["scen"], r["alg"], r["run"]) for r in map(json.loads, open(out_f))}
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[6], t[7]) not in done]
    print(len(tasks), "runs to do", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(out_f, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_run, tasks, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            if i % 300 == 0:
                f.flush(); print(i, r["topology"], r["scen"], r["alg"], round(r["F"], 4), r["feasible"], flush=True)


def analyse():
    import numpy as np
    from scipy.stats import friedmanchisquare, wilcoxon
    R = [json.loads(l) for l in open(OUT / "ablation_shift_runs.jsonl")]
    X13 = [json.loads(l) for l in open(OUT / "shift_search_runs.jsonl")]
    key = lambda r: (r["topology"], r["rho"], r["seed"], r["scen"])
    fref = defaultdict(lambda: np.inf)
    for r in R + X13:
        if r["feasible"]:
            fref[key(r)] = min(fref[key(r)], r["F"])
    out = {}
    for b in [None] + list(CHECK):
        acc = defaultdict(list)
        for r in R:
            i = key(r)
            if not np.isfinite(fref[i]):
                continue
            if b is None:
                acc[(i, r["alg"])].append(r["F"] / fref[i] if r["feasible"] else 2.0)
            elif (r["anytime"] or {}).get(str(b)) is not None:
                acc[(i, r["alg"])].append(r["anytime"][str(b)] / fref[i])
        inst = sorted({i for i, a in acc})
        inst = [i for i in inst if all(acc[(i, a)] for a in VARIANTS)]
        M = np.array([[np.mean(acc[(i, a)]) for a in VARIANTS] for i in inst])
        res = {"n_instances": len(inst), "mean_ratio": {a: float(M[:, j].mean()) for j, a in enumerate(VARIANTS)},
               "best_share": {a: float(np.mean(M[:, j] <= M.min(1) + 1e-12)) for j, a in enumerate(VARIANTS)},
               "friedman_p": float(friedmanchisquare(*M.T).pvalue)}
        pw = []
        for j, a in enumerate(VARIANTS[1:], start=1):
            x, y = M[:, 0], M[:, j]; d = x - y
            p = float(wilcoxon(x, y).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
            pw.append([a, int((d < -1e-12).sum()), int((d > 1e-12).sum()), p, float(np.median(d))])
        order = np.argsort([x[3] for x in pw]); adj = [0.0] * len(pw); run_ = 0.0
        for kk, idx in enumerate(order):
            run_ = max(run_, min(1.0, (len(pw) - kk) * pw[idx][3])); adj[idx] = run_
        res["full_vs"] = {x[0]: {"full_better": x[1], "full_worse": x[2], "median_diff": x[4], "p": x[3], "p_holm": adj[n]}
                          for n, x in enumerate(pw)}
        out["full" if b is None else str(b)] = res
    json.dump(out, open(OUT / "ablation_shift_stats.json", "w"), indent=1)
    for lab, r in out.items():
        print(lab, "n", r["n_instances"], {a: round(x, 3) for a, x in r["mean_ratio"].items()}, "Friedman p", f"{r['friedman_p']:.1e}")
        for a, t in r["full_vs"].items():
            print("    full vs", a, t)


if __name__ == "__main__":
    analyse() if "--analyse" in sys.argv else run(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
