"""Instances, the algorithm registry and the per-run record used by the experiment scripts."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from shield.placement.aco import run_aco
from shield.placement.baselines import (greedy_kcenter, greedy_kmedian, kmeans_placement, pagerank_placement,
                                        run_ga, run_pso, run_random, run_sa)
from shield.placement.common import Evaluator, Result
from shield.placement.eho import run_eho
from shield.placement.exact import ilp_capacitated_kmedian
from shield.placement.hybrid_eho_aco import run_hybrid
from shield.placement.objective import Instance, PlacementProblem, make_demand
from shield.placement.topology import Topology, get_topology, latency_matrix
from shield.utils.io import load_json

# Full memetic hybrid and its component ablations (each switches exactly one part).
HYBRID_VARIANTS = {
    "hybrid_eho_aco": {},
    "hybrid_no_ants": {"separation": "random"},   # EHO restarts instead of ACO ants
    "hybrid_no_ls": {"local_search": "none"},     # no intensification
    "hybrid_swap_ls": {"local_search": "swap"},   # v1's 1-swap search instead of SA intensification
}
STOCHASTIC = [*HYBRID_VARIANTS, "eho", "aco", "ga", "pso", "sa", "random", "kmeans"]
DETERMINISTIC = ["kmedian", "kcenter", "pagerank", "ilp_ckm"]
DEMAND_SEED = 7  # demand pattern is part of the instance, identical for every algorithm and run


def with_params(cfg: dict, params: dict) -> dict:
    """Copy of cfg with {"section.key": value} overrides applied (tuned algorithm settings)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    for dotted, value in params.items():
        section, key = dotted.split(".", 1)
        out[section] = {**out.get(section, {}), key: value}
    return out


def tuned_cfg(cfg: dict, algorithm: str, tuned: dict | None) -> dict:
    """cfg with the algorithm's tuned settings (outputs/placement/tuned_params.json), if any."""
    if not tuned or algorithm not in tuned:
        return cfg
    return with_params(cfg, tuned[algorithm]["params"])


def controller_mu(cfg: dict, mu_flow: float) -> float:
    c = cfg["controller"]
    return c["cores"] * mu_flow * c["eta"]


def detector_mu_flow(latency_file: Path, run_key: str | None, model_key: str, default: float) -> tuple[float, str]:
    """Single-core detector throughput (flows/s, batch 256) from 06_latency.py, else the config default."""
    if latency_file.exists():
        table = load_json(latency_file)
        key = run_key if run_key in table else next((k for k in table if not k.startswith("_")), None)
        entry = table.get(key, {}).get(model_key)
        if entry:
            return float(entry["throughput_b256"]), f"{latency_file.name}:{key}:{model_key}"
    return float(default), "config default"


@lru_cache(maxsize=16)
def topology_and_latency(name: str, cache_dir: str) -> tuple[Topology, np.ndarray]:
    topo = get_topology(name, Path(cache_dir))
    return topo, latency_matrix(topo.G)


def build_problem(topo: Topology, D: np.ndarray, k: int, total_demand: float, mu: float, cfg: dict) -> PlacementProblem:
    lam = make_demand(topo.n, total_demand, cfg["demand_sigma"], DEMAND_SEED)
    inst = Instance(topo.name, D, lam, mu, k, topo.coords)
    o = cfg["objective"]
    return PlacementProblem(inst, o["weights"], cfg["controller"]["u_max"], o["overload_penalty"], o["sla_ms"],
                            o["n_reference"], seed=DEMAND_SEED)


def run_algorithm(name: str, problem: PlacementProblem, topo: Topology, cfg: dict, seed: int,
                  budget: int | None = None) -> Result | None:
    rng = np.random.default_rng(seed)
    k = problem.k
    ev = Evaluator(problem, budget or cfg["budget"])
    t0 = time.time()
    if name in HYBRID_VARIANTS:
        run_hybrid(ev, k, {**cfg["hybrid"], **HYBRID_VARIANTS[name]}, cfg["aco"], rng)
    elif name == "eho":
        run_eho(ev, k, cfg["hybrid"], rng)
    elif name == "aco":
        run_aco(ev, k, cfg["aco"], rng)
    elif name == "ga":
        run_ga(ev, k, cfg["ga"], rng)
    elif name == "pso":
        run_pso(ev, k, cfg["pso"], rng)
    elif name == "sa":
        run_sa(ev, k, cfg["sa"], rng)
    elif name == "random":
        run_random(ev, k, rng)
    elif name == "kmedian":
        ev(greedy_kmedian(problem.D, problem.lam, k))
    elif name == "kcenter":
        ev(greedy_kcenter(problem.D, problem.lam, k))
    elif name == "kmeans":
        ev(kmeans_placement(topo.coords, problem.lam, k, seed))
    elif name == "pagerank":
        ev(pagerank_placement(topo.G, k))
    elif name == "ilp_ckm":
        C = ilp_capacitated_kmedian(problem, cfg["ilp_time_limit"])
        if C is None:
            return None
        ev(C)
    else:
        raise ValueError(f"Unknown algorithm {name!r}")
    res = ev.result()
    res.seconds = time.time() - t0
    return res


def compress_history(history: list[tuple[int, float]], budget: int, points: int = 50) -> list[float]:
    """Best-so-far F at `points` evenly spaced evaluation counts (for convergence plots)."""
    if not history:
        return []
    grid = np.linspace(1, budget, points)
    evals = np.array([h[0] for h in history])
    vals = np.array([h[1] for h in history])
    idx = np.searchsorted(evals, grid, side="right") - 1
    return [float(vals[i]) if i >= 0 else float("nan") for i in idx]
