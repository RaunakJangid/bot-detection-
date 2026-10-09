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


def _tri(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership; a == b or b == c gives a shoulder."""
    if x <= a:
        return 1.0 if a == b else 0.0
    if x >= c:
        return 1.0 if b == c else 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def adaptive_control(stagnation: float, diversity: float, mode: str) -> float:
    """Exploration pressure in [0, 1] from stagnation (0 = improving, 1 = stalled for stall_gens) and clan
    diversity (0 = all elephants identical, 1 = fully diverse). v3 diagnosis: the hybrid converges
    prematurely (clan diversity -> k/n, pheromone entropy -> its MMAS floor, best F frozen early).
    "fuzzy": 4 Sugeno rules (stalled AND converged -> 1; improving AND diverse -> 0; mixed -> 0.5);
    "linear": stagnation * (1 - diversity), the plain-rule control for the fuzzy variant."""
    if mode == "linear":
        return stagnation * (1.0 - diversity)
    s_hi, s_lo = _tri(stagnation, 0.3, 1.0, 1.0), _tri(stagnation, 0.0, 0.0, 0.7)
    d_lo, d_hi = _tri(diversity, 0.0, 0.0, 0.5), _tri(diversity, 0.2, 1.0, 1.0)
    rules = [(min(s_hi, d_lo), 1.0), (min(s_hi, d_hi), 0.5), (min(s_lo, d_lo), 0.5), (min(s_lo, d_hi), 0.0)]
    w = sum(r for r, _ in rules)
    return sum(r * o for r, o in rules) / w if w > 0 else 0.0


def run_hybrid(ev: Evaluator, k: int, cfg: dict, aco_cfg: dict, rng: np.random.Generator) -> None:
    p = ev.problem
    colony = Colony(p.D, p.lam, aco_cfg, rng)
    near = nearest_lists(p.D)
    if cfg.get("separation", "ants") == "ants":
        separator = lambda: colony.construct(k)
    else:
        separator = lambda: random_solution(p.n, k, rng)
    ls = cfg.get("local_search", "sa")
    mode = cfg.get("adaptive", "none")
    stall_gens, x = cfg.get("stall_gens", 10), 0.0
    gen_cfg, last_best, stalled = cfg, np.inf, 0
    try:
        clans = init_clans(ev, k, cfg["n_clans"], cfg["clan_size"], rng, separator)
        gen = 0
        while not ev.exhausted:
            eho_generation(ev, clans, k, gen_cfg, rng, separator)
            gen += 1
            matriarchs = [min(clan, key=lambda e: e[1]) for clan in clans]
            if mode != "none":   # adapt evaporation and the number of ant-rebuilt elephants per clan
                stalled = 0 if ev.best_F < last_best - 1e-12 else stalled + 1
                last_best = min(last_best, ev.best_F)
                used = len(set(np.concatenate([C for clan in clans for C, _ in clan]).tolist()))
                div = (used - k) / max(1, min(p.n, k * sum(len(c) for c in clans)) - k)
                x = adaptive_control(min(1.0, stalled / stall_gens), min(1.0, max(0.0, div)), mode)
                gen_cfg = {**cfg, "ants_per_clan": max(cfg.get("ants_per_clan", 1),
                                                         int(round(x * (cfg["clan_size"] - 1))))}
            colony.update(matriarchs + [(ev.best, ev.best_F)], ev.best_F)
            if mode != "none" and x > 0:
                # MMAS pheromone trail smoothing (Stutzle & Hoos 2000): pull trails toward tau_max so ants
                # explore again. (Raising evaporation would do the opposite in MMAS: faster convergence.)
                colony.tau += x * cfg.get("smooth", 0.5) * (colony.tau_max - colony.tau)
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
