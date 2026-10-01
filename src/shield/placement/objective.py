"""The resilient, capacity-aware controller placement objective.

Given controller set C (k node ids):
  * capacitated greedy-regret assignment gives every switch a primary controller and a backup
    (the nearest other controller);
  * latency(v) = propagation D[v, primary] + M/M/1 sojourn 1000 / (mu - load) ms;
  * R_fail = worst, over single-controller failures, of the mean switch latency after failover.

F = w_avg*avg + w_max*max + w_inter*inter + w_imb*CoV(util) + w_fail*R_fail   (each / random-placement mean)
    + overload_penalty * (excess utilisation, normal + worst failure)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

TERMS = ("avg", "max", "inter", "imbalance", "fail")
EXTRA = ("overload", "fail_overload", "pct_sla", "fail_pct_sla")


@njit(cache=True)
def assign_capacitated(Dsub, lam, cap):
    """Greedy regret assignment. Switches with the largest (2nd nearest - nearest) gap choose first and
    take the nearest controller with room; if none has room, the nearest one (overload)."""
    n, k = Dsub.shape
    primary = np.zeros(n, np.int64)
    backup = np.zeros(n, np.int64)
    if k == 1:
        return primary, backup
    # Per-switch controller order by latency (insertion sort: k is small).
    rank = np.empty((n, k), np.int64)
    neg_regret = np.empty(n)
    for v in range(n):
        for c in range(k):
            j = c
            while j > 0 and Dsub[v, rank[v, j - 1]] > Dsub[v, c]:
                rank[v, j] = rank[v, j - 1]
                j -= 1
            rank[v, j] = c
        neg_regret[v] = Dsub[v, rank[v, 0]] - Dsub[v, rank[v, 1]]
    order = np.argsort(neg_regret)
    remaining = np.full(k, cap)
    for v in order:
        chosen = rank[v, 0]
        for j in range(k):
            if remaining[rank[v, j]] >= lam[v]:
                chosen = rank[v, j]
                break
        remaining[chosen] -= lam[v]
        primary[v] = chosen
        backup[v] = rank[v, 1] if rank[v, 0] == chosen else rank[v, 0]
    return primary, backup


@njit(cache=True)
def _latencies(Dsub, lam, assign, mu, cap, k):
    n = Dsub.shape[0]
    loads = np.zeros(k)
    for v in range(n):
        loads[assign[v]] += lam[v]
    wait = np.empty(k)
    for c in range(k):
        wait[c] = 1000.0 / (mu - min(loads[c], cap))
    lat = np.empty(n)
    for v in range(n):
        lat[v] = Dsub[v, assign[v]] + wait[assign[v]]
    return lat, loads


@njit(cache=True)
def placement_terms(Dsub, Dcc, lam, mu, cap, sla, fail_k1):
    """Returns [avg, max, inter, imbalance, fail, overload, fail_overload, pct_sla, fail_pct_sla]."""
    n, k = Dsub.shape
    u_max = cap / mu
    primary, backup = assign_capacitated(Dsub, lam, cap)
    lat, loads = _latencies(Dsub, lam, primary, mu, cap, k)
    out = np.zeros(9)
    out[0] = lat.mean()
    out[1] = lat.max()
    if k > 1:
        s = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                s += Dcc[i, j]
        out[2] = s / (k * (k - 1) / 2)
    util = loads / mu
    m = util.mean()
    out[3] = util.std() / m if m > 0 else 0.0
    over = 0.0
    for c in range(k):
        over += max(0.0, util[c] - u_max)
    out[5] = over
    out[7] = np.sum(lat <= sla) / n
    if k == 1:
        out[4] = fail_k1
        out[6] = 0.0
        out[8] = 0.0
        return out
    worst, worst_over, worst_pct = 0.0, 0.0, 1.0
    alt = np.empty(n, np.int64)
    for f in range(k):
        for v in range(n):
            alt[v] = backup[v] if primary[v] == f else primary[v]
        lat_f, loads_f = _latencies(Dsub, lam, alt, mu, cap, k)
        a = lat_f.mean()
        if a > worst:
            worst = a
        o = 0.0
        for c in range(k):
            if c != f:
                o += max(0.0, loads_f[c] / mu - u_max)
        if o > worst_over:
            worst_over = o
        p = np.sum(lat_f <= sla) / n
        if p < worst_pct:
            worst_pct = p
    out[4] = worst
    out[6] = worst_over
    out[8] = worst_pct
    return out


def make_demand(n: int, total: float, sigma: float, seed: int) -> np.ndarray:
    lam = np.random.default_rng(seed).lognormal(0.0, sigma, n)
    return lam / lam.sum() * total


@dataclass
class Instance:
    name: str
    D: np.ndarray          # (n, n) shortest-path latency, ms
    lam: np.ndarray        # (n,) demand, flows/s
    mu: float              # per-controller service rate, flows/s
    k: int
    coords: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.lam)


class PlacementProblem:
    def __init__(self, inst: Instance, weights: dict, u_max: float, overload_penalty: float, sla_ms: float,
                 n_reference: int = 200, seed: int = 0):
        self.inst = inst
        self.D = np.ascontiguousarray(inst.D, dtype=np.float64)
        self.lam = np.ascontiguousarray(inst.lam, dtype=np.float64)
        self.mu, self.k, self.n = float(inst.mu), inst.k, inst.n
        self.cap = u_max * self.mu
        self.sla = sla_ms
        self.w = np.array([weights[t] for t in TERMS])
        self.overload_penalty = overload_penalty
        self.fail_k1 = 2.0 * float(self.D.max())
        rng = np.random.default_rng(seed)
        samples = np.array([self.raw(np.sort(rng.choice(self.n, self.k, replace=False)))[:5]
                            for _ in range(n_reference)])
        ref = samples.mean(0)
        self.ref = np.where(ref > 1e-12, ref, 1.0)

    def raw(self, C: np.ndarray) -> np.ndarray:
        C = np.asarray(C, dtype=np.int64)
        Dsub = np.ascontiguousarray(self.D[:, C])
        Dcc = np.ascontiguousarray(self.D[np.ix_(C, C)])
        return placement_terms(Dsub, Dcc, self.lam, self.mu, self.cap, self.sla, self.fail_k1)

    def score(self, r: np.ndarray) -> float:
        return float((self.w * r[:5] / self.ref).sum() + self.overload_penalty * (r[5] + r[6]))

    def __call__(self, C: np.ndarray) -> float:
        return self.score(self.raw(C))

    def describe(self, C: np.ndarray) -> dict:
        r = self.raw(C)
        d = {t: float(r[i]) for i, t in enumerate(TERMS + EXTRA)}
        d["F"] = self.score(r)
        return d
