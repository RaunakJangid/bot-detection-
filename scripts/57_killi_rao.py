"""P2 fairness, faithful version: the exact FFCCP of Killi & Rao (IEEE Commun. Lett. 20(6), 2016), on their instances.

Model (their eqs. 1-11): place p controllers; every switch has K reference controllers, the l-th being its l-th
closest controller (closest-assignment constraints 6-7); a controller's load is the sum of the loads of all switches
for which it is any of the K references (eq. 8, one shared capacity U); objective = worst-case latency from a switch
to its K-th reference controller (eq. 1). With closest assignment the assignment is determined by the controller
set, so a candidate set is scored exactly: z(C) = max_i d(i, K-th closest in C), feasible iff every controller's
load <= U (infeasible sets get a penalty above any feasible value).
Instances (their Sec. IV): AT&T (AttMpls, 25 nodes), GEANT (40 sites), SURFNET (50 nodes) from the Topology Zoo;
latency = great-circle distance from latitude/longitude at 2e5 km/s; switch loads U(100k, 400k) PACKET_IN/s
(seeded; theirs are random too, so values differ); U = 7,800 kpps; K = 2 (one failure) and K = 3 (two failures);
p from the minimum feasible (ceil(K * sum L / U)) to their largest p.
Exact optimum: their ILP (eqs. 1-11) with scipy HiGHS (time limit 30 min). Metaheuristics: random, ga, pso, sa,
gwo, gwo_sa, hybrid_eho_aco, hybrid_linear, v3 tuned settings, 10 runs x 20,000 evaluations each."""

import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import networkx as nx
import numpy as np
from numba import njit
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shield.cli import parser
from shield.placement.runner import run_algorithm, tuned_cfg
from shield.placement.topology import download_zoo, haversine_km
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, load_json, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("killirao")
ALGS = ["random", "ga", "pso", "sa", "gwo", "gwo_sa", "hybrid_eho_aco", "hybrid_linear"]
TOPOS = {"AttMpls": {2: 8, 3: 9}, "Geant2012": {2: 9, 3: 11}, "Surfnet": {2: 10, 3: 12}}   # their largest p
U, RUNS, BUDGET = 7800.0, 10, 20_000


def instance(name, paths):
    f = download_zoo(name, paths.topologies)
    G = nx.read_graphml(f) if str(f).endswith("graphml") else nx.read_gml(f, label="id")
    pts = sorted({(round(float(d["Latitude"]), 5), round(float(d["Longitude"]), 5))
                  for _, d in G.nodes(data=True) if "Latitude" in d and "Longitude" in d})
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = haversine_km(*pts[i], *pts[j]) / 200.0          # ms at 2e5 km/s
    L = np.random.default_rng(1000 + n).uniform(100, 400, n)                   # kpps
    return D, L


@njit
def ffccp_z(D, L, C, K, U):
    n, p = D.shape[0], len(C)
    load = np.zeros(p)
    z = 0.0
    dd = np.empty(p)
    for i in range(n):
        for j in range(p):
            dd[j] = D[i, C[j]]
        order = np.argsort(dd)
        for l in range(K):
            load[order[l]] += L[i]
        if dd[order[K - 1]] > z:
            z = dd[order[K - 1]]
    over = 0.0
    for j in range(p):
        if load[j] > U:
            over += load[j] - U
    if over > 0:
        return 1e3 + over
    return z


class FFCCP:
    def __init__(self, D, L, k, K):
        self.D, self.lam, self.L, self.n, self.k, self.K = D, L, L, len(D), k, K

    def __call__(self, C):
        return float(ffccp_z(self.D, self.L, np.asarray(C, np.int64), self.K, U))


def milp_ffccp(D, L, p, K, time_limit=1800):
    """Killi & Rao ILP: y_j, r^l_ij (l = 1..K), z."""
    n = len(D)
    iy = lambda j: j
    ir = lambda l, i, j: n + (l * n + i) * n + j
    iz = n + K * n * n
    nv = iz + 1
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(cs, vs, a, b):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs); lo.append(a); hi.append(b); r += 1

    ub = np.ones(nv); ub[iz] = np.inf
    for i in range(n):
        for l in range(K):
            add([ir(l, i, j) for j in range(n)], [1] * n, 1, 1)                      # (2), (3)
            if l > 0:
                ub[ir(l, i, i)] = 0                                                  # l-th ref != i for l > 1
    add([iy(j) for j in range(n)], [1] * n, p, p)                                    # (4)
    for i in range(n):
        for j in range(n):
            add([ir(l, i, j) for l in range(K)] + [iy(j)], [1] * K + [-1], -np.inf, 0)   # (5)
            closer = np.flatnonzero(D[i] <= D[i, j])
            for l in range(K):                                                       # (6), (7)
                cs = [ir(l, i, a) for a in closer] + [ir(m, i, j) for m in range(l)] + [iy(j)]
                add(cs, [1] * (len(closer) + l) + [-1], 0, np.inf)
        add([ir(K - 1, i, j) for j in range(n)] + [iz], list(D[i]) + [-1], -np.inf, 0)    # (9)
    for j in range(n):                                                               # (8)
        add([ir(l, i, j) for l in range(K) for i in range(n)] + [iy(j)], [L[i] for l in range(K) for i in range(n)] + [-U],
            -np.inf, 0)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv))
    c = np.zeros(nv); c[iz] = 1
    integ = np.ones(nv); integ[iz] = 0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None
    return {"z": float(res.fun), "status": int(res.status), "C": np.flatnonzero(res.x[:n] > 0.5).tolist(),
            "bound": float(getattr(res, "mip_dual_bound", float("nan")) or float("nan"))}


_W = {}


def _init():
    _W["pcfg"] = load_config("placement", False)
    _W["paths"] = Paths(False)
    tf = _W["paths"].v3 / "placement" / "tuned_params.json"
    _W["tuned"] = load_json(tf) if tf.exists() else None


class _Topo:            # run_algorithm only needs coords/G for kmeans/pagerank, which we do not run
    pass


def _run(task):
    name, K, p, alg, seed = task
    D, L = instance(name, _W["paths"])
    prob = FFCCP(D, L, p, K)
    t = time.time()
    res = run_algorithm(alg, prob, _Topo(), tuned_cfg(_W["pcfg"], alg, _W["tuned"]), seed, budget=BUDGET)
    return {"topology": name, "n": len(D), "K": K, "p": p, "alg": alg, "seed": seed, "C": res.best.tolist(),
            "z": res.best_F, "seconds": time.time() - t}


def main():
    p_ = parser(__doc__, datasets=False)
    p_.add_argument("--workers", type=int, default=12)
    args = p_.parse_args()
    P = Paths(False)
    od = P.v3 / "killi_rao"; od.mkdir(parents=True, exist_ok=True)
    grid = []
    for name, pmax in TOPOS.items():
        D, L = instance(name, P)
        for K, pm in pmax.items():
            pmin = max(K, math.ceil(K * L.sum() / U))
            grid += [(name, K, p) for p in range(pmin, pm + 1)]
        log.info("%s: %d sites, total load %.0f kpps", name, len(D), L.sum())
    runs_f = od / "runs.jsonl"
    done = {(r["topology"], r["K"], r["p"], r["alg"], r["seed"]) for r in read_jsonl(runs_f)}
    tasks = [(t, K, p, a, s) for t, K, p in grid for a in ALGS for s in range(RUNS) if (t, K, p, a, s) not in done]
    log.info("%d instances, %d runs to do", len(grid), len(tasks))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for row in pool.imap_unordered(_run, tasks, chunksize=4):
            append_jsonl(row, runs_f)
    opt_f = od / "optimum.json"
    opt = json.loads(opt_f.read_text()) if opt_f.exists() else {}
    for t, K, p in grid:
        key = f"{t}|{K}|{p}"
        if key in opt:
            continue
        D, L = instance(t, P)
        tt = time.time()
        r = milp_ffccp(D, L, p, K)
        opt[key] = {**r, "seconds": time.time() - tt} if r else None
        log.info("optimum %s: %s", key, opt[key])
        save_json(opt, opt_f)


if __name__ == "__main__":
    main()
