"""P2 extensions on the faithful Killi & Rao FFCCP model (scripts/57_killi_rao.py).

--part shift   Escalation shift on the PUBLISHED model. Controllers run the student->teacher cascade, so a controller's
               capacity scales with the cascade throughput: U(e) = 7800 * mu(e) / mu(0.10), mu(e) = 1/(1/mu_s + e/mu_t)
               (single-core throughputs from outputs/latency/latency.json), i.e. the network is sized as in the paper
               when the planned 10% escalation holds. Scenarios e = measured test escalation at a 10% validation
               budget (outputs_v3/escalation_shift/scenarios_b10.csv). AT&T, GEANT, SURFNET; K = 2. Planned placement
               = ILP optimum at the minimum feasible p for e = 0.10. Per scenario: is the planned placement still
               within capacity (closest-assignment loads), and the ILP optimum at the minimum feasible p(e).
--part psosa   Hybrid PSO-SA (Mohanty & Sahoo 2024 compare GA, PSO, PSO-SA, GWO; their full text was unavailable, so
               the standard form is used: set-based PSO for 70% of the budget, then SA intensification from the
               best solution for the rest) on the 43 faithful instances; appended to killi_rao/runs.jsonl.
--part large   Larger networks where methods can differ: Cogentco (197) and Kdl (754 sites), K = 2, loads
               U(100, 400) kpps, U = 7800, p from the minimum feasible to +5; all methods incl. pso_sa; 10 runs x
               20,000 evaluations; reference = best value found (no proven optimum at this size)."""

import importlib.util
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shield.cli import parser
from shield.placement.common import BudgetExhausted, Evaluator, nearest_lists
from shield.placement.baselines import run_pso
from shield.placement.hybrid_eho_aco import sa_intensify
from shield.placement.runner import run_algorithm, tuned_cfg
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, load_json, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("ffccp_ext")
_s = importlib.util.spec_from_file_location("kr", Path(__file__).with_name("57_killi_rao.py"))
kr = importlib.util.module_from_spec(_s); _s.loader.exec_module(kr)
ALGS = kr.ALGS + ["pso_sa"]
LARGE = ["Cogentco", "Kdl"]


class _Limited:
    """Evaluator proxy that stops a phase after `limit` evaluations (the shared budget keeps counting)."""

    def __init__(self, ev, limit):
        self.ev, self.limit, self.start = ev, limit, ev.n_evals
        self.problem = ev.problem

    def __call__(self, C):
        if self.ev.n_evals - self.start >= self.limit:
            raise BudgetExhausted
        return self.ev(C)

    @property
    def best(self):
        return self.ev.best

    @property
    def best_F(self):
        return self.ev.best_F


def run_pso_sa(prob, cfg, seed, budget):
    rng = np.random.default_rng(seed)
    ev = Evaluator(prob, budget)
    t = time.time()
    run_pso(_Limited(ev, int(0.7 * budget)), prob.k, cfg["pso"], rng)
    try:
        sa_intensify(ev.best, ev.best_F, ev, nearest_lists(prob.D), ev.remaining, cfg["gwo_sa"]["t0_frac"], rng)
    except BudgetExhausted:
        pass
    return ev.best, ev.best_F, time.time() - t


_W = {}


def _init():
    kr._init()
    _W.update(kr._W)


def _run(task):
    name, K, p, alg, seed = task
    D, L = kr.instance(name, _W["paths"])
    prob = kr.FFCCP(D, L, p, K)
    if alg == "pso_sa":
        cfg = tuned_cfg(tuned_cfg(_W["pcfg"], "pso", _W["tuned"]), "gwo_sa", _W["tuned"])
        best, F, sec = run_pso_sa(prob, cfg, seed, kr.BUDGET)
    else:
        t = time.time()
        res = run_algorithm(alg, prob, kr._Topo(), tuned_cfg(_W["pcfg"], alg, _W["tuned"]), seed, budget=kr.BUDGET)
        best, F, sec = res.best, res.best_F, time.time() - t
    return {"topology": name, "n": len(D), "K": K, "p": p, "alg": alg, "seed": seed, "C": [int(c) for c in best],
            "z": float(F), "seconds": sec}


def pool_run(tasks, out_f, workers):
    done = {(r["topology"], r["K"], r["p"], r["alg"], r["seed"]) for r in read_jsonl(out_f)}
    tasks = [t for t in tasks if t not in done]
    log.info("%d runs to do -> %s", len(tasks), out_f)
    if not tasks:
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for row in pool.imap_unordered(_run, tasks, chunksize=2):
            append_jsonl(row, out_f)


def loads_feasible(D, L, C, K, U):
    C = np.asarray(C)
    load = np.zeros(len(C))
    for i in range(len(D)):
        order = np.argsort(D[i, C])
        load[order[:K]] += L[i]
    return bool((load <= U + 1e-9).all()), float(load.max() / U)


def part_shift(P):
    od = P.v3 / "killi_rao"; out_f = od / "shift.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    ent = load_json(P.outputs / "latency" / "latency.json")["ciciot_class34"]
    mu_s, mu_t = ent["shield"]["throughput_b256"], ent["teacher"]["throughput_b256"]
    mu = lambda e: 1.0 / (1.0 / mu_s + e / mu_t)
    s10 = pd.read_csv(P.v3 / "escalation_shift" / "scenarios_b10.csv")
    scen = sorted({round(float(x), 2) for x in s10.e_test} | {0.10})
    K = 2
    for name in kr.TOPOS:
        if name in res:
            continue
        D, L = kr.instance(name, P)
        Uof = lambda e: kr.U * mu(e) / mu(0.10)
        p0 = max(K, math.ceil(K * L.sum() / Uof(0.10)))
        plan = None
        while plan is None:
            kr.U, U_save = Uof(0.10), kr.U
            plan = kr.milp_ffccp(D, L, p0, K, time_limit=600)
            kr.U = U_save
            if plan is None:
                p0 += 1
        rows = []
        for e in scen:
            U = Uof(e)
            ok, util = loads_feasible(D, L, plan["C"], K, U)
            p = max(K, math.ceil(K * L.sum() / U))
            best = None
            while best is None and p <= len(D):
                kr.U, U_save = U, kr.U
                best = kr.milp_ffccp(D, L, p, K, time_limit=600)
                kr.U = U_save
                if best is None:
                    p += 1
            rows.append({"e": e, "capacity_kpps": U, "planned_p": p0, "planned_z": plan["z"], "planned_feasible": ok,
                         "planned_max_util": util, "p_min": p, "z_at_p_min": best["z"] if best else None,
                         "status": best["status"] if best else None})
            log.info("%s e=%.2f U=%.0f planned(p=%d) feasible=%s util=%.2f | p_min=%d z=%.2f", name, e, U, p0, ok,
                     util, p, best["z"] if best else float("nan"))
        res[name] = rows
        save_json(res, out_f)


def main():
    a = parser(__doc__, datasets=False)
    a.add_argument("--part", choices=["shift", "psosa", "large"], required=True)
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--runs", type=int, default=kr.RUNS, help="seeds per (setting, method) for --part large")
    args = a.parse_args()
    P = Paths(False)
    od = P.v3 / "killi_rao"; od.mkdir(parents=True, exist_ok=True)
    if args.part == "shift":
        part_shift(P)
    elif args.part == "psosa":
        grid = []
        for name, pmax in kr.TOPOS.items():
            D, L = kr.instance(name, P)
            for K, pm in pmax.items():
                grid += [(name, K, p) for p in range(max(K, math.ceil(K * L.sum() / kr.U)), pm + 1)]
        pool_run([(t, K, p, "pso_sa", s) for t, K, p in grid for s in range(kr.RUNS)], od / "runs.jsonl", args.workers)
    else:
        tasks = []
        for name in LARGE:
            D, L = kr.instance(name, P)
            pmin = max(2, math.ceil(2 * L.sum() / kr.U))
            log.info("%s: %d sites, load %.0f, p %d..%d", name, len(D), L.sum(), pmin, pmin + 5)
            tasks += [(name, 2, p, alg, s) for p in range(pmin, pmin + 6) for alg in ALGS for s in range(args.runs)]
        pool_run(tasks, od / "large_runs.jsonl", args.workers)


if __name__ == "__main__":
    main()
