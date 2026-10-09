"""Further population metaheuristics (v3), discretised for k-subsets with the same set operators as
PSO/EHO (`mix`, `swap_neighbor`, slot sampling) and run on the same evaluation budget.

- GWO  Grey Wolf Optimizer (Mirjalili et al., 2014): each slot of a wolf's next position is drawn from
       the alpha/beta/delta wolves; with probability a/2 (a: 2 -> 0 over the budget) a slot explores
       (random node). `gwo_sa` adds the same SA intensification of the best as the memetic hybrid, as in
       the GWO + SA hybrids reported strongest for controller placement (e.g. GEWO, Khojand et al. 2023).
- FOA  Fruit Fly Optimization (Pan, 2012): flies search around the swarm location (`radius` neighbour
       swaps, shrinking over the budget); the swarm flies to the best fly when it is better.
- WOA  Whale Optimization (Mirjalili & Lewis, 2016): encircling / searching (move toward the best or a
       random whale with strength 1 - |A|) or a bubble-net spiral (move toward the best + local swap).
- HHO  Harris Hawks Optimization (Heidari et al., 2019): exploration (perch on a random hawk / random
       node) while |E| >= 1, soft/hard besiege toward the rabbit (best) with greedy rapid dives after.
- DE   Differential Evolution (Storn & Price, 1997), set version: donor = a + F * (b \\ c), binomial
       crossover with the target at rate CR, a neighbour-swap mutation with probability `mut` (keeps
       diversity in the set encoding), greedy selection. A non-metaphor reference.
"""

from __future__ import annotations

import numpy as np

from shield.placement.common import (BudgetExhausted, Evaluator, fill, mix, nearest_lists, random_solution,
                                     swap_neighbor)
from shield.placement.hybrid_eho_aco import sa_intensify


def _progress(ev: Evaluator) -> float:
    return min(1.0, ev.n_evals / ev.budget)


def _pick(sources: list[np.ndarray | None], probs: np.ndarray, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Slot sampling: each slot from one source (None = random node), duplicates resolved by `fill`."""
    chosen: set[int] = set()
    for _ in range(4 * k):
        if len(chosen) >= k:
            break
        src = sources[rng.choice(len(sources), p=probs)]
        chosen.add(int(rng.integers(n)) if src is None else int(src[rng.integers(len(src))]))
    return fill(chosen, k, n, rng)


def run_gwo(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator, ls_cfg: dict | None = None) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D) if ls_cfg else None
    try:
        pack = [(C, ev(C)) for C in (random_solution(n, k, rng) for _ in range(cfg["pack"]))]
        it = 0
        while True:
            pack.sort(key=lambda w: w[1])
            leaders = [pack[0][0], pack[min(1, len(pack) - 1)][0], pack[min(2, len(pack) - 1)][0]]
            a = 2.0 * (1.0 - _progress(ev))
            explore = min(1.0, cfg["explore"] * a / 2.0)
            probs = np.array([*([(1 - explore) / 3] * 3), explore])
            new = []
            for C, F in pack:
                X = _pick([*leaders, None], probs, k, n, rng)
                FX = ev(X)
                new.append((X, FX) if FX <= F or not cfg.get("greedy", True) else (C, F))
            pack = new
            it += 1
            if ls_cfg and it % ls_cfg["every"] == 0:
                best, best_F = sa_intensify(ev.best, ev.best_F, ev, near, ls_cfg["evals"], ls_cfg["t0_frac"], rng)
                pack.sort(key=lambda w: w[1])
                pack[-1] = (best, best_F)
    except BudgetExhausted:
        pass


def run_foa(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)
    try:
        loc = random_solution(n, k, rng)
        loc_F = ev(loc)
        while True:
            r = max(1, int(round(cfg["radius"] * (1.0 - _progress(ev)) * k)))
            flies = []
            for _ in range(cfg["flies"]):
                X = loc
                for _ in range(rng.integers(1, r + 1)):
                    X = swap_neighbor(X, n, near, rng, p_local=cfg["p_local"])
                flies.append((X, ev(X)))
            X, FX = min(flies, key=lambda f: f[1])
            if FX < loc_F:
                loc, loc_F = X, FX
    except BudgetExhausted:
        pass


def run_woa(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)
    try:
        pod = [(C, ev(C)) for C in (random_solution(n, k, rng) for _ in range(cfg["pod"]))]
        while True:
            a = 2.0 * (1.0 - _progress(ev))
            for i, (C, F) in enumerate(pod):
                A = abs(2 * a * rng.random() - a)
                if rng.random() < 0.5:
                    guide = ev.best if A < 1 else pod[rng.integers(len(pod))][0]
                    X = mix(C, guide, max(0.05, 1.0 - A / 2.0), rng)
                else:   # bubble-net spiral: close in on the best, then a local move around it
                    X = swap_neighbor(mix(C, ev.best, cfg["spiral"], rng), n, near, rng)
                FX = ev(X)
                if FX <= F:
                    pod[i] = (X, FX)
    except BudgetExhausted:
        pass


def run_hho(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)
    try:
        hawks = [(C, ev(C)) for C in (random_solution(n, k, rng) for _ in range(cfg["hawks"]))]
        while True:
            for i, (C, F) in enumerate(hawks):
                E = 2.0 * (2 * rng.random() - 1) * (1.0 - _progress(ev))   # escaping energy
                if abs(E) >= 1:   # exploration
                    X = (mix(C, hawks[rng.integers(len(hawks))][0], 0.5, rng) if rng.random() < 0.5
                         else _pick([C, None], np.array([1 - cfg["perch"], cfg["perch"]]), k, n, rng))
                elif abs(E) >= 0.5:   # soft besiege: partial move toward the rabbit (best)
                    X = mix(C, ev.best, 1.0 - abs(E), rng)
                else:                 # hard besiege: a position next to the rabbit, not on it
                    X = swap_neighbor(ev.best, n, near, rng)
                FX = ev(X)
                if abs(E) < 1:        # progressive rapid dives: greedy local moves from X
                    for _ in range(cfg["dives"]):
                        Y = swap_neighbor(X, n, near, rng)
                        FY = ev(Y)
                        if FY < FX:
                            X, FX = Y, FY
                if FX <= F:
                    hawks[i] = (X, FX)
    except BudgetExhausted:
        pass


def run_de(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)
    try:
        pop = [(C, ev(C)) for C in (random_solution(n, k, rng) for _ in range(cfg["pop"]))]
        while True:
            for i, (C, F) in enumerate(pop):
                a, b, c = (pop[j][0] for j in rng.choice([j for j in range(len(pop)) if j != i], 3, replace=False))
                donor = a.tolist()
                in_c, in_a = set(c.tolist()), set(donor)
                diff = [int(v) for v in b if v not in in_c and v not in in_a]
                rng.shuffle(diff)
                for j in range(k):   # differential step: each slot replaced with prob F by a b \ c node
                    if diff and rng.random() < cfg["F"]:
                        donor[j] = diff.pop()
                in_donor = set(donor)
                trial = C.tolist()
                pool = [v for v in donor if v not in set(trial)]
                rng.shuffle(pool)
                jrand = rng.integers(k)
                for j in range(k):   # binomial crossover: target slots not in the donor take donor nodes
                    if pool and (rng.random() < cfg["CR"] or j == jrand) and trial[j] not in in_donor:
                        trial[j] = pool.pop()
                X = fill(set(trial), k, n, rng)
                if rng.random() < cfg["mut"]:   # keeps diversity once the population has converged
                    X = swap_neighbor(X, n, near, rng)
                FX = ev(X)
                if FX <= F:
                    pop[i] = (X, FX)
    except BudgetExhausted:
        pass
