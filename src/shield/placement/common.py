"""Budgeted evaluation and set-based search operators shared by every placement algorithm.

A solution is a sorted int array of k distinct node ids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from shield.placement.objective import PlacementProblem


class BudgetExhausted(Exception):
    pass


@dataclass
class Result:
    best: np.ndarray
    best_F: float
    n_evals: int
    seconds: float
    history: list[tuple[int, float]] = field(default_factory=list)  # (evaluations, best F) at improvements


class Evaluator:
    """Counts every objective call against a fixed budget (cache hits included, so all algorithms pay
    the same price per proposal) and records the best-so-far trajectory."""

    def __init__(self, problem: PlacementProblem, budget: int):
        self.problem, self.budget = problem, budget
        self.n_evals = 0
        self.best, self.best_F = None, np.inf
        self.history: list[tuple[int, float]] = []
        self._cache: dict[tuple, float] = {}
        self._t0 = time.time()

    @property
    def exhausted(self) -> bool:
        return self.n_evals >= self.budget

    @property
    def remaining(self) -> int:
        return self.budget - self.n_evals

    def __call__(self, C: np.ndarray) -> float:
        if self.exhausted:
            raise BudgetExhausted
        self.n_evals += 1
        key = tuple(C.tolist())
        F = self._cache.get(key)
        if F is None:
            F = self.problem(C)
            self._cache[key] = F
        if F < self.best_F:
            self.best_F, self.best = F, C.copy()
            self.history.append((self.n_evals, F))
        return F

    def result(self) -> Result:
        return Result(self.best, float(self.best_F), self.n_evals, time.time() - self._t0, self.history)


def random_solution(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(n, k, replace=False))


def nearest_lists(D: np.ndarray, m: int = 10) -> np.ndarray:
    """For each node, its m nearest other nodes by latency."""
    m = min(m, len(D) - 1)
    return np.argsort(D, axis=1)[:, 1:m + 1]


def swap_neighbor(C: np.ndarray, n: int, near: np.ndarray, rng: np.random.Generator,
                  p_local: float = 0.5) -> np.ndarray:
    """Replace one controller: by one of its graph-near nodes (prob p_local) or by a random node."""
    if len(C) >= n:
        return C.copy()
    members = set(C.tolist())
    i = rng.integers(len(C))
    candidates = [int(v) for v in near[C[i]] if v not in members] if rng.random() < p_local else []
    if candidates:
        new = candidates[rng.integers(len(candidates))]
    else:
        new = int(rng.integers(n))
        while new in members:
            new = int(rng.integers(n))
    out = C.copy()
    out[i] = new
    return np.sort(out)


def mix(base: np.ndarray, guide: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Discrete 'move toward guide': each base node absent from guide is replaced, with probability p,
    by a guide node not yet present."""
    out = set(base.tolist())
    pool = [int(g) for g in guide if g not in out]
    rng.shuffle(pool)
    for b in base:
        if not pool:
            break
        if b not in guide and rng.random() < p:
            out.discard(int(b))
            out.add(pool.pop())
    return np.sort(np.fromiter(out, dtype=np.int64, count=len(out)))


def consensus(solutions: list[np.ndarray], k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """The k most frequent nodes across a group of solutions (random tie-break)."""
    counts = np.bincount(np.concatenate(solutions), minlength=n).astype(float)
    counts += rng.random(n) * 1e-3
    return np.sort(np.argsort(-counts)[:k])


def fill(chosen: set[int], k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    while len(chosen) < k:
        chosen.add(int(rng.integers(n)))
    return np.sort(np.fromiter(chosen, dtype=np.int64, count=len(chosen)))


def one_swap_local_search(C: np.ndarray, F: float, ev: Evaluator, near: np.ndarray, max_evals: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """First-improvement 1-swap search over graph-near candidates, capped at max_evals."""
    used = 0
    improved = True
    while improved and used < max_evals and not ev.exhausted:
        improved = False
        members = set(C.tolist())
        for i in rng.permutation(len(C)):
            for v in near[C[i]]:
                if v in members:
                    continue
                cand = C.copy()
                cand[i] = v
                cand = np.sort(cand)
                Fc = ev(cand)
                used += 1
                if Fc < F:
                    C, F, improved = cand, Fc, True
                    break
                if used >= max_evals or ev.exhausted:
                    return C, F
            if improved:
                break
    return C, F
