"""Hybrid EHO-ACO (memetic).

* EHO clan structure does exploitation (elephants follow matriarchs, matriarchs follow consensus).
* Separating operator: the worst elephant of each clan is rebuilt by an ACO ant ("ants"), so restarts
  come from a learned distribution (pheromone) instead of uniformly at random ("random" = plain EHO).
* Pheromone is reinforced by every clan matriarch and the global best each generation (MMAS bounds).
* Intensification every `local_search_every` generations on the global best:
    "sa"   - a short simulated-annealing run (neighbour swaps, geometric cooling), the memetic step;
    "swap" - first-improvement 1-swap search (the original v1 design);
    "none" - no local search.
  The improved solution is fed back into the worst clan and reinforces the pheromone.
Component ablations (runner.HYBRID_VARIANTS) switch exactly one of these parts.
"""

from __future__ import annotations

import math

import numpy as np

from shield.placement.aco import Colony
from shield.placement.common import (BudgetExhausted, Evaluator, nearest_lists, one_swap_local_search,
                                     random_solution, swap_neighbor)
from shield.placement.eho import eho_generation, init_clans


def sa_intensify(C: np.ndarray, F: float, ev: Evaluator, near: np.ndarray, evals: int, t0_frac: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Simulated annealing from (C, F) for at most `evals` evaluations; returns the best visited.
    Temperature starts at t0_frac * |F| and cools geometrically to 1% of that by the end."""
    n = ev.problem.n
    T = max(t0_frac * abs(F), 1e-9)
    cooling = 0.01 ** (1.0 / max(evals, 1))
    cur, cur_F, best, best_F = C, F, C, F
    for _ in range(evals):
        if ev.exhausted:
            break
        cand = swap_neighbor(cur, n, near, rng)
        Fc = ev(cand)
        if Fc <= cur_F or rng.random() < math.exp(-(Fc - cur_F) / T):
            cur, cur_F = cand, Fc
            if Fc < best_F:
                best, best_F = cand, Fc
        T *= cooling
    return best, best_F


def run_hybrid(ev: Evaluator, k: int, cfg: dict, aco_cfg: dict, rng: np.random.Generator) -> None:
    p = ev.problem
    colony = Colony(p.D, p.lam, aco_cfg, rng)
    near = nearest_lists(p.D)
    if cfg.get("separation", "ants") == "ants":
        separator = lambda: colony.construct(k)
    else:
        separator = lambda: random_solution(p.n, k, rng)
    ls = cfg.get("local_search", "sa")
    try:
        clans = init_clans(ev, k, cfg["n_clans"], cfg["clan_size"], rng, separator)
        gen = 0
        while not ev.exhausted:
            eho_generation(ev, clans, k, cfg, rng, separator)
            gen += 1
            matriarchs = [min(clan, key=lambda e: e[1]) for clan in clans]
            colony.update(matriarchs + [(ev.best, ev.best_F)], ev.best_F)
            if ls != "none" and gen % cfg["local_search_every"] == 0:
                if ls == "sa":
                    best, best_F = sa_intensify(ev.best, ev.best_F, ev, near, cfg["local_search_evals"],
                                                cfg.get("sa_t0_frac", 0.05), rng)
                else:
                    best, best_F = one_swap_local_search(ev.best, ev.best_F, ev, near, cfg["local_search_evals"], rng)
                # Feed the improved solution back into the clan whose matriarch is worst.
                worst_clan = max(range(len(clans)), key=lambda i: min(e[1] for e in clans[i]))
                clans[worst_clan].sort(key=lambda e: e[1])
                clans[worst_clan][-1] = (best, best_F)
    except BudgetExhausted:
        pass
