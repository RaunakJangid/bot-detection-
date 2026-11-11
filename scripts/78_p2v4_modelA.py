"""P2 v4, experiments X4 (Model A rerun) and X9 (planning with headroom). CPU only. Output: outputs_v5/p2/modelA_v4.jsonl.

Model A of scripts/54 (topologies Geant2012, Cogentco, syn200; rho in {0.5, 0.7, 0.9}; total demand = rho x 4 x mu_c(e0);
relative SLA of configs/coupled.yaml; feasible = 95% of switches within the SLA with no overload, normally and after the
worst single-controller failure), with three changes:
  capacity   per scenario from the measured cost of the cascade that produced it (X1, outputs_v5/p2/capacity_v4.json):
             mu_c(scenario) = r x mu_c(e0), r = capacity ratio (batch 256 and batch 1), plus a grid of ratios; the plan
             is r = 1 (e0 = 10%);
  search     for every k: SA, EHO-ACO and GWO+SA (tuned settings), 6 runs x 40,000 evaluations each (v3: GWO+SA,
             3 x 10,000); k is feasible if ANY run's best placement is feasible; k is scanned upward from the capacity
             lower bound ceil(4 rho / r) (at least 2, because of the failure requirement);
  instances  5 demand draws per topology and rho (seeds 7, 107, 207, 307, 407; seed 7 is the v3 instance).
k_min(r) is reported raw and as the monotone envelope: a placement feasible at a lower capacity is also feasible at a
higher one, so k_env(r) = min over r' <= r of k_raw(r').
X9: every placement found at k_min(r_plan) is re-evaluated at every scenario ratio r: feasible?, % within SLA, overload.
Usage: 78_p2v4_modelA.py [workers]"""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2"); OUT.mkdir(parents=True, exist_ok=True)
ALGS, RUNS, BUDGET, K_MAX = ("sa", "hybrid_eho_aco", "gwo_sa"), 6, 40_000, 30
SEEDS = (7, 107, 207, 307, 407)
_W = {}


def ratios():
    cap = json.load(open(OUT / "capacity_v4.json"))["modelb_ratio"]
    r = {"plan": 1.0, **{k: float(v) for k, v in cap.items()}}
    for g in (0.9, 0.7, 0.6, 0.5):
        r[f"grid_{g:.2f}"] = g
    return dict(sorted(r.items(), key=lambda kv: -kv[1]))


def _init():
    from shield.utils.config import Paths, load_config
    from shield.utils.io import load_json
    _W["P"] = Paths(False)
    _W["pcfg"], _W["ccfg"] = load_config("placement", False), load_config("coupled", False)
    _W["tuned"] = load_json(_W["P"].v3 / "placement" / "tuned_params.json")
    _W["mu0"] = json.load(open("D:/shield_run/outputs_v3/escalation_shift/meta.json"))["mu_controller"]["0.1"]


def _task(t):
    from shield.placement.coupling import feasible, topology_sla
    from shield.placement.objective import make_demand
    from shield.placement.objective import Instance, PlacementProblem
    from shield.placement.runner import run_algorithm, topology_and_latency, tuned_cfg, with_params
    topo_name, rho, seed, R = t
    P, pcfg, ccfg, tuned, mu0 = _W["P"], _W["pcfg"], _W["ccfg"], _W["tuned"], _W["mu0"]
    topo, D = topology_and_latency(topo_name, str(P.topologies))
    lam = make_demand(topo.n, 1.0, pcfg["demand_sigma"], seed)
    sla = topology_sla(D, lam, ccfg.get("sla", {}), ccfg["k_ref"], pcfg["objective"]["sla_ms"])
    cfgs = {a: with_params(tuned_cfg(pcfg, a, tuned), {"objective.sla_ms": sla}) for a in ALGS}
    total = rho * ccfg["k_ref"] * mu0
    t0 = time.time()

    def problem(k, r, alg):
        # as runner.build_problem, but with this instance's demand draw (build_problem fixes DEMAND_SEED)
        o = cfgs[alg]["objective"]
        inst = Instance(topo.name, D, make_demand(topo.n, total, cfgs[alg]["demand_sigma"], seed), r * mu0, k, topo.coords)
        return PlacementProblem(inst, o["weights"], cfgs[alg]["controller"]["u_max"], o["overload_penalty"], o["sla_ms"],
                                o["n_reference"], seed=seed)

    def solve(k, r):
        best, feas = None, None
        for alg in ALGS:
            pr = problem(k, r, alg)
            for s in range(RUNS):
                res = run_algorithm(alg, pr, topo, cfgs[alg], s, budget=BUDGET)
                d = pr.describe(res.best)
                if feasible(d, k, ccfg) and (feas is None or res.best_F < feas[1]):
                    feas = (np.asarray(res.best).tolist(), float(res.best_F), alg, s)
                if best is None or res.best_F < best[1]:
                    best = (np.asarray(res.best).tolist(), float(res.best_F), alg, s)
        return feas, best

    out = {"topology": topo_name, "n": topo.n, "rho": rho, "seed": seed, "sla_ms": float(sla), "ratios": R, "k_raw": {}, "place": {}}
    for name, r in R.items():
        k = max(2, math.ceil(ccfg["k_ref"] * rho / r - 1e-9))
        found = None
        while k <= min(K_MAX, topo.n - 1):
            feas, _ = solve(k, r)
            if feas is not None:
                found = (k, feas); break
            k += 1
        out["k_raw"][name] = found[0] if found else None
        if found:
            out["place"][name] = {"k": found[0], "C": found[1][0], "F": found[1][1], "alg": found[1][2]}
    # monotone envelope (ratios sorted descending): a placement feasible at r' is feasible at any r >= r'
    names = list(R)
    out["k_env"] = {}
    for i, name in enumerate(names):
        cands = [out["k_raw"][m] for m in names[i:] if out["k_raw"][m] is not None]
        out["k_env"][name] = min(cands) if cands else None
    # X9: re-evaluate each planned placement at every scenario capacity (objective config of SA; SLA identical)
    out["cross"] = {}
    for pn, pl in out["place"].items():
        row = {}
        for sn, r in R.items():
            pr = problem(pl["k"], r, "sa")
            d = pr.describe(np.asarray(pl["C"]))
            row[sn] = {"feasible": bool(feasible(d, pl["k"], ccfg)), "pct_sla": float(d["pct_sla"]),
                       "fail_pct_sla": float(d["fail_pct_sla"]), "overload": float(d["overload"]),
                       "fail_overload": float(d.get("fail_overload", float("nan")))}
        out["cross"][pn] = row
    out["seconds"] = time.time() - t0
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    R = ratios()
    from shield.utils.config import load_config
    ccfg = load_config("coupled", False)
    out_f = OUT / "modelA_v4.jsonl"
    done = set()
    if out_f.exists():
        for ln in open(out_f):
            r = json.loads(ln); done.add((r["topology"], r["rho"], r["seed"]))
    tasks = [(t, float(rho), s, R) for t in ccfg["topologies"] for rho in ccfg["rho_values"] for s in SEEDS
             if (t, float(rho), s) not in done]
    tasks.sort(key=lambda x: (-x[1], x[0] != "Geant2012"))          # heaviest (high rho) first
    print(f"{len(tasks)} tasks, ratios {R}", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for res in pool.imap_unordered(_task, tasks):
            with open(out_f, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(res["topology"], res["rho"], res["seed"], f"{res['seconds'] / 60:.1f} min", res["k_env"], flush=True)


if __name__ == "__main__":
    main()
