"""Hybrid EHO-ACO.

* EHO clan structure does exploitation (elephants follow matriarchs, matriarchs follow consensus).
* EHO's random separating operator is replaced by ACO ants, so restarts are drawn from a learned
  distribution (pheromone) instead of uniformly at random.
* Pheromone is reinforced by every clan matriarch and the global best each generation (MMAS bounds).
* Every `local_search_every` generations the global best gets a bounded 1-swap local search.
"""

from __future__ import annotations

import numpy as np

from shield.placement.aco import Colony
from shield.placement.common import BudgetExhausted, Evaluator, nearest_lists, one_swap_local_search
from shield.placement.eho import eho_generation, init_clans


def run_hybrid(ev: Evaluator, k: int, cfg: dict, aco_cfg: dict, rng: np.random.Generator) -> None:
    p = ev.problem
    colony = Colony(p.D, p.lam, aco_cfg, rng)
    near = nearest_lists(p.D)
    ant = lambda: colony.construct(k)
    try:
        clans = init_clans(ev, k, cfg["n_clans"], cfg["clan_size"], rng, ant)
        gen = 0
        while not ev.exhausted:
            eho_generation(ev, clans, k, cfg, rng, ant)
            gen += 1
            matriarchs = [min(clan, key=lambda e: e[1]) for clan in clans]
            colony.update(matriarchs + [(ev.best, ev.best_F)], ev.best_F)
            if gen % cfg["local_search_every"] == 0:
                best, best_F = one_swap_local_search(ev.best, ev.best_F, ev, near, cfg["local_search_evals"], rng)
                # Seed the improved solution back into the clan whose matriarch is worst.
                worst_clan = max(range(len(clans)), key=lambda i: min(e[1] for e in clans[i]))
                clans[worst_clan].sort(key=lambda e: e[1])
                clans[worst_clan][-1] = (best, best_F)
    except BudgetExhausted:
        pass
