"""P2 v4, experiment X13: search methods under escalation shift (Model A), with per-run results. CPU only.

Closes the open question left by X4: X4 kept only the best placement per (instance, scenario, k) and credited exact ties
to the first method in its order (SA), so the method comparison under shift had a tie bias and no test.
Instances: Model A of scripts/78 (Geant2012, Cogentco, syn200; rho in {0.5, 0.7, 0.9}; demand draws 7 and 107) at three
scenarios (16% CICIoMT2024 6-class, 27% IoT-23 1-1, 13% online cap; batch-256 capacity ratios), each at its tight
controller count k = k_env from modelA_v4.jsonl (the smallest k found feasible); instances with no feasible k are skipped.
Methods (tuned settings of scripts/78): SA, EHO-ACO (Algorithm 1), GWO+SA, GA, PSO. 30 seeds x 20,000 evaluations.
Per run: best objective F, feasibility of the best placement (95% within SLA, no overload, normally and after the worst
single failure), best F after 1k, 2k, 5k, 10k and 20k evaluations where the method exposes its history.
Output: outputs_v5/p2/shift_search_runs.jsonl; analysis in scripts/95."""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
ALGS = ("sa", "hybrid_eho_aco", "gwo_sa", "ga", "pso")
SEEDS, BUDGET, CHECK = 30, 20_000, (1000, 2000, 5000, 10000, 20000)
SCEN = ("b256_e16_ciciomt6", "b256_e27_iot23_1-1", "b256_e13_online_cap_ciciomt6")
DRAWS = (7, 107)
_W = {}


def _init():
    from shield.utils.config import Paths, load_config
    from shield.utils.io import load_json
    _W["P"] = Paths(False)
    _W["pcfg"], _W["ccfg"] = load_config("placement", False), load_config("coupled", False)
    _W["tuned"] = load_json(_W["P"].v3 / "placement" / "tuned_params.json")
    _W["mu0"] = json.load(open("D:/shield_run/outputs_v3/escalation_shift/meta.json"))["mu_controller"]["0.1"]


def _run(t):
    from shield.placement.coupling import feasible, topology_sla
    from shield.placement.objective import Instance, PlacementProblem, make_demand
    from shield.placement.runner import run_algorithm, topology_and_latency, tuned_cfg, with_params
    topo_name, rho, seed, scen, k, r, alg, s = t
    P, pcfg, ccfg, tuned, mu0 = _W["P"], _W["pcfg"], _W["ccfg"], _W["tuned"], _W["mu0"]
    topo, D = topology_and_latency(topo_name, str(P.topologies))
    lam = make_demand(topo.n, 1.0, pcfg["demand_sigma"], seed)
    sla = topology_sla(D, lam, ccfg.get("sla", {}), ccfg["k_ref"], pcfg["objective"]["sla_ms"])
    cfg = with_params(tuned_cfg(pcfg, alg, tuned), {"objective.sla_ms": sla})
    o = cfg["objective"]
    inst = Instance(topo.name, D, make_demand(topo.n, rho * ccfg["k_ref"] * mu0, cfg["demand_sigma"], seed), r * mu0, k,
                    topo.coords)
    pr = PlacementProblem(inst, o["weights"], cfg["controller"]["u_max"], o["overload_penalty"], o["sla_ms"],
                          o["n_reference"], seed=seed)
    res = run_algorithm(alg, pr, topo, cfg, s, budget=BUDGET)
    d = pr.describe(res.best)
    hist = getattr(res, "history", None); any_ = None
    if hist:
        any_ = {str(c): (float(min(f for e, f in hist if e <= c)) if any(e <= c for e, f in hist) else None) for c in CHECK}
    return {"topology": topo_name, "rho": rho, "seed": seed, "scen": scen, "k": k, "r": r, "alg": alg, "run": s,
            "F": float(res.best_F), "feasible": bool(feasible(d, k, ccfg)), "C": np.asarray(res.best).tolist(),
            "anytime": any_}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    global DRAWS
    if len(sys.argv) > 2:                     # optional: demand draws to run, e.g. 207,307,407
        DRAWS = tuple(int(x) for x in sys.argv[2].split(","))
    rows = [json.loads(l) for l in open(OUT / "modelA_v4.jsonl")]
    tasks = []
    for row in rows:
        if row["seed"] not in DRAWS:
            continue
        for sc in SCEN:
            k = row["k_env"].get(sc)
            if k is None:
                continue
            for a in ALGS:
                for s in range(SEEDS):
                    tasks.append((row["topology"], row["rho"], row["seed"], sc, int(k), float(row["ratios"][sc]), a, s))
    out_f = OUT / "shift_search_runs.jsonl"
    done = set()
    if out_f.exists():
        for ln in open(out_f):
            x = json.loads(ln); done.add((x["topology"], x["rho"], x["seed"], x["scen"], x["alg"], x["run"]))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[6], t[7]) not in done]
    print(len(tasks), "runs to do", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(out_f, "a") as f:
        for i, x in enumerate(pool.imap_unordered(_run, tasks, chunksize=2)):
            f.write(json.dumps(x) + "\n")
            if i % 200 == 0:
                f.flush(); print(i, x["topology"], x["rho"], x["scen"], x["alg"], round(x["F"], 4), x["feasible"], flush=True)


if __name__ == "__main__":
    main()
