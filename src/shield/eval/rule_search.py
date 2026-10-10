"""Black-box optimisers over integer vectors x in {0..G-1}^D, maximising f(x) (a validation macro-F1), with an
exact evaluation budget so every method gets the same effort. Used to tune detection decision rules (per-class
logit biases) and ensemble weights (v3 step: metaheuristics applied to the detector itself).

  random, coord (coordinate ascent = the existing rule fitter's search), sa, pso, gwo_sa (GWO + SA polish, the
  published hybrid that won the placement benchmark), eho_aco (ours: EHO clans + MMAS ant construction + SA
  intensification + linear stagnation/diversity control, the hybrid_linear design)."""

from __future__ import annotations

import numpy as np


class Budget(Exception):
    pass


class Objective:
    def __init__(self, f, budget: int):
        self.f, self.budget, self.n = f, budget, 0
        self.best_x, self.best_f, self.trace = None, -np.inf, []

    def __call__(self, x: np.ndarray) -> float:
        if self.n >= self.budget:
            raise Budget
        self.n += 1
        v = float(self.f(x))
        if v > self.best_f:
            self.best_f, self.best_x = v, x.copy()
        self.trace.append(self.best_f)
        return v


def _sa(obj, x, fx, G, rng, evals, t0=2e-3, t1=1e-5):
    for i in range(evals):
        t = t0 * (t1 / t0) ** (i / max(1, evals - 1))
        y = x.copy(); d = rng.integers(len(x))
        y[d] = np.clip(y[d] + rng.choice([-3, -2, -1, 1, 2, 3]), 0, G - 1)
        fy = obj(y)
        if fy >= fx or rng.random() < np.exp((fy - fx) / t):
            x, fx = y, fy
    return x, fx


def random_search(obj, D, G, x0, rng):
    obj(x0)
    while True:
        obj(rng.integers(0, G, D))


def coord(obj, D, G, x0, rng):
    x, fx = x0.copy(), obj(x0)
    while True:
        for d in rng.permutation(D):
            for g in range(G):
                if g == x[d]:
                    continue
                y = x.copy(); y[d] = g
                fy = obj(y)
                if fy > fx:
                    x, fx = y, fy


def sa(obj, D, G, x0, rng):
    _sa(obj, x0.copy(), obj(x0), G, rng, obj.budget)


def pso(obj, D, G, x0, rng, n=20, w=0.7, c1=1.5, c2=1.5):
    X = np.vstack([x0, rng.uniform(0, G - 1, (n - 1, D))]).astype(float)
    V = np.zeros_like(X)
    P, Pf = X.copy(), np.array([obj(np.rint(x).astype(int)) for x in X])
    while True:
        g = P[Pf.argmax()]
        V = w * V + c1 * rng.random(X.shape) * (P - X) + c2 * rng.random(X.shape) * (g - X)
        X = np.clip(X + V, 0, G - 1)
        for i in range(n):
            f = obj(np.rint(X[i]).astype(int))
            if f > Pf[i]:
                P[i], Pf[i] = X[i], f


def gwo_sa(obj, D, G, x0, rng, n=20, polish=0.3):
    X = np.vstack([x0, rng.uniform(0, G - 1, (n - 1, D))]).astype(float)
    F = np.array([obj(np.rint(x).astype(int)) for x in X])
    stop = int(obj.budget * (1 - polish))
    while obj.n < stop:
        a = 2 * (1 - obj.n / stop)
        lead = X[np.argsort(-F)[:3]]
        new = np.zeros_like(X)
        for L in lead:
            A = a * (2 * rng.random(X.shape) - 1); C = 2 * rng.random(X.shape)
            new += L - A * np.abs(C * L - X)
        X = np.clip(new / 3, 0, G - 1)
        F = np.array([obj(np.rint(x).astype(int)) for x in X])
    _sa(obj, obj.best_x.copy(), obj.best_f, G, rng, obj.budget)


def eho_aco(obj, D, G, x0, rng, eta=None, n_clans=5, clan_size=8, alpha=0.5, beta=0.1, rho=0.1,
            ls_every=2, ls_evals=50, stall_gens=10, smooth=0.5):
    eta = np.ones(G) if eta is None else eta
    tau = np.ones((D, G)); tmax, tmin = 1.0, 1.0 / (2 * G)

    def ant():
        p = tau * eta; p /= p.sum(1, keepdims=True)
        return (p.cumsum(1) > rng.random((D, 1))).argmax(1)

    clans = [[(x0.copy(), obj(x0))] + [(a, obj(a)) for a in (ant() for _ in range(clan_size - 1))]]
    for _ in range(n_clans - 1):
        clans.append([(a, obj(a)) for a in (ant() for _ in range(clan_size))])
    gen, stalled, last, ctl = 0, 0, -np.inf, 0.0
    while True:
        gen += 1
        for c, clan in enumerate(clans):
            clan.sort(key=lambda e: -e[1])
            m = clan[0][0]
            centre = np.mean([e[0] for e in clan], 0)
            upd = [(np.clip(np.rint(m + beta * (centre - m)), 0, G - 1).astype(int))]
            for x, _ in clan[1:]:
                upd.append(np.clip(np.rint(x + alpha * rng.random(D) * (m - x)), 0, G - 1).astype(int))
            n_ants = max(1, int(round(ctl * (clan_size - 1))))     # separation: worst members rebuilt by ants
            for j in range(1, n_ants + 1):
                upd[-j] = ant()
            clans[c] = [(x, obj(x)) for x in upd]
        improved = obj.best_f > last + 1e-12
        stalled = 0 if improved else stalled + 1
        last = max(last, obj.best_f)
        used = np.mean([len(np.unique([e[0][d] for cl in clans for e in cl])) for d in range(D)])
        div = (used - 1) / max(1, min(G, n_clans * clan_size) - 1)
        ctl = min(1.0, stalled / stall_gens) * (1 - min(1.0, div))    # linear control (hybrid_linear)
        tau *= 1 - rho
        tau[np.arange(D), obj.best_x] += rho * tmax
        tau = np.clip(tau, tmin, tmax)
        if ctl > 0:
            tau += ctl * smooth * (tmax - tau)                         # MMAS trail smoothing
        if gen % ls_every == 0:
            x, fx = _sa(obj, obj.best_x.copy(), obj.best_f, G, rng, ls_evals)
            worst = min(range(n_clans), key=lambda i: max(e[1] for e in clans[i]))
            clans[worst].sort(key=lambda e: -e[1]); clans[worst][-1] = (x, fx)


METHODS = {"random": random_search, "coord": coord, "sa": sa, "pso": pso, "gwo_sa": gwo_sa, "eho_aco": eho_aco}


def run(name, f, D, G, x0, budget, seed, **kw):
    obj = Objective(f, budget)
    try:
        METHODS[name](obj, D, G, x0, np.random.default_rng(seed), **kw)
    except Budget:
        pass
    return obj
