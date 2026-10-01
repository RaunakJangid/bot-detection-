"""Ant Colony Optimisation for k-node placement (MAX-MIN Ant System pheromone bounds)."""

from __future__ import annotations

import numpy as np

from shield.placement.common import BudgetExhausted, Evaluator


class Colony:
    """Ants pick nodes one at a time with probability ~ tau^a * eta^b.

    eta: demand-weighted closeness (low mean latency to the demand = attractive).
    Coverage mask: after a pick, nodes within `radius` of it are skipped, so one ant's controllers
    spread out; the mask is relaxed when it would leave nothing to choose.
    """

    def __init__(self, D: np.ndarray, lam: np.ndarray, cfg: dict, rng: np.random.Generator):
        self.D, self.n, self.rng = D, len(D), rng
        self.a, self.b, self.rho = cfg["a"], cfg["b"], cfg["evaporation"]
        mean_lat = (lam[:, None] * D).sum(0) / lam.sum()
        eta = 1.0 / (mean_lat + 1e-9)
        self.eta_b = (eta / eta.max()) ** self.b
        off_diag = D[~np.eye(self.n, dtype=bool)]
        self.radius = float(np.quantile(off_diag[np.isfinite(off_diag)], cfg["mask_quantile"]))
        self.tau_max, self.tau_min = 1.0, 1.0 / (2 * self.n)
        self.tau = np.full(self.n, self.tau_max)

    def construct(self, k: int) -> np.ndarray:
        weight = self.tau ** self.a * self.eta_b
        allowed = np.ones(self.n, dtype=bool)
        free = np.ones(self.n, dtype=bool)
        chosen = []
        for _ in range(k):
            p = weight * allowed
            if p.sum() <= 0:
                p = weight * free
            v = int(self.rng.choice(self.n, p=p / p.sum()))
            chosen.append(v)
            free[v] = False
            allowed &= free & (self.D[v] >= self.radius)
        return np.sort(np.array(chosen, dtype=np.int64))

    def update(self, deposits: list[tuple[np.ndarray, float]], best_F: float) -> None:
        self.tau *= 1.0 - self.rho
        for C, F in deposits:
            self.tau[C] += 1.0 / max(F, 1e-9)
        self.tau_max = 1.0 / (self.rho * max(best_F, 1e-9))
        self.tau_min = self.tau_max / (2 * self.n)
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)


def run_aco(ev: Evaluator, k: int, cfg: dict, rng: np.random.Generator) -> None:
    p = ev.problem
    colony = Colony(p.D, p.lam, cfg, rng)
    try:
        while not ev.exhausted:
            ants = [colony.construct(k) for _ in range(min(cfg["n_ants"], ev.remaining))]
            scored = [(C, ev(C)) for C in ants]
            it_best = min(scored, key=lambda t: t[1])
            colony.update([it_best, (ev.best, ev.best_F)], ev.best_F)
    except BudgetExhausted:
        pass
