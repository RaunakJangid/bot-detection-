"""Exact references: brute force over all k-subsets (true optimum of our objective, small cases only)
and a capacitated k-median ILP (optimal for propagation latency alone), solved with HiGHS via SciPy."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shield.placement.objective import PlacementProblem


def brute_force(problem: PlacementProblem, limit: int) -> dict | None:
    n, k = problem.n, problem.k
    total = math.comb(n, k)
    if total > limit:
        return None
    t0 = time.time()
    best, best_F = None, np.inf
    for combo in itertools.combinations(range(n), k):
        C = np.array(combo, dtype=np.int64)
        F = problem(C)
        if F < best_F:
            best, best_F = C, F
    return {"best": best, "best_F": float(best_F), "evaluated": total, "seconds": time.time() - t0}


def _ckm_constraints(D: np.ndarray, lam: np.ndarray, k: int, cap: float | None) -> LinearConstraint:
    """Variables: x[v, c] at v*n + c (assignment, continuous in [0, 1]), then y[c] at n*n + c (open, binary)."""
    n = len(D)
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    vv, cc = np.divmod(np.arange(n * n), n)
    # each switch assigned exactly once
    rows.append(vv); cols.append(np.arange(n * n)); vals.append(np.ones(n * n))
    lb += [1.0] * n; ub += [1.0] * n; r += n
    # x[v, c] <= y[c]
    rows += [r + np.arange(n * n), r + np.arange(n * n)]
    cols += [np.arange(n * n), n * n + cc]
    vals += [np.ones(n * n), -np.ones(n * n)]
    lb += [-np.inf] * (n * n); ub += [0.0] * (n * n); r += n * n
    # exactly k controllers
    rows.append(np.full(n, r)); cols.append(n * n + np.arange(n)); vals.append(np.ones(n))
    lb.append(k); ub.append(k); r += 1
    if cap is not None:  # sum_v lam_v x[v, c] <= cap * y[c]
        rows += [r + cc, r + np.arange(n)]
        cols += [np.arange(n * n), n * n + np.arange(n)]
        vals += [lam[vv], -np.full(n, cap)]
        lb += [-np.inf] * n; ub += [0.0] * n; r += n
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n * n + n))
    return LinearConstraint(A.tocsr(), lb, ub)


def ilp_capacitated_kmedian(problem: PlacementProblem, time_limit: int, max_nodes: int = 200) -> np.ndarray | None:
    """min sum_v sum_c x_vc D_vc  s.t. each switch assigned once, to an open controller, k open,
    controller load <= cap. Falls back to the uncapacitated model if capacity makes it infeasible."""
    n, k, D, lam = problem.n, problem.k, problem.D, problem.lam
    if n > max_nodes:
        return None
    c = np.concatenate([D.ravel(), np.zeros(n)])
    integrality = np.concatenate([np.zeros(n * n), np.ones(n)])
    for cap in (problem.cap, None):
        res = milp(c, constraints=_ckm_constraints(D, lam, k, cap), integrality=integrality,
                   bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
        if res.x is not None:
            opened = np.flatnonzero(res.x[n * n:] > 0.5)
            if len(opened) == k:
                return np.sort(opened.astype(np.int64))
    return None
