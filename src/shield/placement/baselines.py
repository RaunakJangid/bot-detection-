"""Baseline placement methods: constructive heuristics and metaheuristics on the same budget."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from shield.placement.common import (BudgetExhausted, Evaluator, fill, nearest_lists, random_solution,
                                     swap_neighbor)

# ---------------------------------------------------------------- constructive heuristics


def greedy_kmedian(D: np.ndarray, lam: np.ndarray, k: int) -> np.ndarray:
    """Add, one at a time, the node that most reduces demand-weighted latency to the nearest controller."""
    best = np.full(len(D), np.inf)
    chosen = []
    for _ in range(k):
        cost = (lam[:, None] * np.minimum(best[:, None], D)).sum(0)
        cost[chosen] = np.inf
        j = int(np.argmin(cost))
        chosen.append(j)
        best = np.minimum(best, D[:, j])
    return np.sort(np.array(chosen))


def greedy_kcenter(D: np.ndarray, lam: np.ndarray, k: int) -> np.ndarray:
    """Farthest-first traversal starting from the 1-median."""
    chosen = [int(np.argmin((lam[:, None] * D).sum(0)))]
    dist = D[:, chosen[0]].copy()
    while len(chosen) < k:
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, D[:, j])
    return np.sort(np.array(chosen))


def kmeans_placement(coords: np.ndarray, lam: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(k, n_init=10, random_state=seed).fit(coords, sample_weight=lam)
    chosen: set[int] = set()
    for c in km.cluster_centers_:
        order = np.argsort(np.linalg.norm(coords - c, axis=1))
        chosen.add(int(next(v for v in order if v not in chosen)))
    return np.sort(np.fromiter(chosen, dtype=np.int64))


def pagerank_placement(G: nx.Graph, k: int) -> np.ndarray:
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["w"] = 1.0 / d["delay"]
    pr = nx.pagerank(H, weight="w")
    return np.sort(np.array(sorted(pr, key=pr.get, reverse=True)[:k]))


# ---------------------------------------------------------------- metaheuristics


def run_random(ev: Evaluator, k: int, rng: np.random.Generator) -> None:
    try:
        while True:
            ev(random_solution(ev.problem.n, k, rng))
    except BudgetExhausted:
        pass


def _crossover(a: np.ndarray, b: np.ndarray, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    common = set(a.tolist()) & set(b.tolist())
    rest = list((set(a.tolist()) | set(b.tolist())) - common)
    rng.shuffle(rest)
    child = set(common)
    while len(child) < k and rest:
        child.add(rest.pop())
    return fill(child, k, n, rng)


def run_ga(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)

    def tournament(pop):
        idx = rng.choice(len(pop), cfg["tournament"], replace=False)
        return min((pop[i] for i in idx), key=lambda e: e[1])[0]

    try:
        pop = []
        for _ in range(cfg["pop"]):
            C = random_solution(n, k, rng)
            pop.append((C, ev(C)))
        while True:
            pop.sort(key=lambda e: e[1])
            nxt = pop[:cfg["elite"]]
            while len(nxt) < cfg["pop"]:
                a, b = tournament(pop), tournament(pop)
                child = _crossover(a, b, k, n, rng) if rng.random() < cfg["cx_rate"] else a.copy()
                if rng.random() < cfg["mut_rate"]:
                    child = swap_neighbor(child, n, near, rng)
                nxt.append((child, ev(child)))
            pop = nxt
    except BudgetExhausted:
        pass


def run_pso(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    """Set-based discrete PSO: each slot of a particle's next position is drawn from its current
    position (w), personal best (c1), global best (c2) or a random node (remainder)."""
    n = ev.problem.n
    probs = np.array([cfg["w"], cfg["c1"], cfg["c2"]])
    probs = np.append(probs, max(0.0, 1.0 - probs.sum()))
    probs /= probs.sum()
    try:
        X = [random_solution(n, k, rng) for _ in range(cfg["swarm"])]
        P = [(x, ev(x)) for x in X]
        while True:
            for i in range(len(X)):
                sources = [X[i], P[i][0], ev.best, None]
                chosen: set[int] = set()
                for _ in range(4 * k):
                    if len(chosen) >= k:
                        break
                    src = sources[rng.choice(4, p=probs)]
                    v = int(rng.integers(n)) if src is None else int(src[rng.integers(len(src))])
                    chosen.add(v)
                X[i] = fill(chosen, k, n, rng)
                F = ev(X[i])
                if F < P[i][1]:
                    P[i] = (X[i], F)
    except BudgetExhausted:
        pass


def run_sa(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    n = ev.problem.n
    near = nearest_lists(ev.problem.D)
    try:
        samples = [ev(random_solution(n, k, rng)) for _ in range(cfg["t0_samples"])]
        T = max(float(np.std(samples)), 1e-6)
        C = ev.best.copy()
        F = ev.best_F
        while True:
            cand = swap_neighbor(C, n, near, rng)
            Fc = ev(cand)
            if Fc <= F or rng.random() < math.exp(-(Fc - F) / T):
                C, F = cand, Fc
            T = max(T * cfg["cooling"], 1e-9)
    except BudgetExhausted:
        pass
