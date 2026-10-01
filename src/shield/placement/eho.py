"""Elephant Herding Optimisation, discretised for k-subsets.

Clan updating: each elephant moves toward its clan's matriarch (best member) via `mix(.., alpha)`;
the matriarch moves toward the clan consensus via `mix(.., beta)`. Each clan's matriarch keeps the
better of its old and new position (elitism). Separating: the worst elephant of each clan is replaced
by a fresh solution from `separator` (random in pure EHO, an ACO ant in the hybrid).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from shield.placement.common import BudgetExhausted, Evaluator, consensus, mix, random_solution

Elephant = tuple[np.ndarray, float]


def init_clans(ev: Evaluator, k: int, n_clans: int, clan_size: int, rng: np.random.Generator,
               make: Callable[[], np.ndarray]) -> list[list[Elephant]]:
    clans = []
    for _ in range(n_clans):
        clan = []
        for _ in range(clan_size):
            C = make()
            clan.append((C, ev(C)))
        clans.append(clan)
    return clans


def eho_generation(ev: Evaluator, clans: list[list[Elephant]], k: int, cfg: dict, rng: np.random.Generator,
                   separator: Callable[[], np.ndarray]) -> None:
    n = ev.problem.n
    for ci, clan in enumerate(clans):
        clan.sort(key=lambda e: e[1])
        matriarch = clan[0]
        centre = consensus([C for C, _ in clan], k, n, rng)
        new_clan = []
        M = mix(matriarch[0], centre, cfg["beta"], rng)
        FM = ev(M)
        new_clan.append((M, FM) if FM < matriarch[1] else matriarch)
        for C, _ in clan[1:]:
            X = mix(C, matriarch[0], cfg["alpha"], rng)
            new_clan.append((X, ev(X)))
        new_clan.sort(key=lambda e: e[1])
        for j in range(1, min(cfg.get("ants_per_clan", 1), len(new_clan) - 1) + 1):
            S = separator()
            new_clan[-j] = (S, ev(S))
        clans[ci] = new_clan


def run_eho(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    make = lambda: random_solution(n, k, rng)
    try:
        clans = init_clans(ev, k, cfg["n_clans"], cfg["clan_size"], rng, make)
        while not ev.exhausted:
            eho_generation(ev, clans, k, cfg, rng, make)
    except BudgetExhausted:
        pass
