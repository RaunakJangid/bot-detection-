"""P2 fairness check: our placement optimisers on the PUBLISHED problem model, with exact optima.

Model: failure-foresight capacitated controller placement (FFCCPP; Killi & Rao, IEEE 2016), the model used by
Mohanty & Sahoo (CCPE 2024, e8254: enhanced GA, PSO, hybrid PSO-SA and GWO on four Topology Zoo networks) and their
SA paper (JKSU-CIS 2021). Each switch i (load L_i) gets a primary and a distinct backup controller; per-level
capacity sum_i L_i x_ij^r <= U for r in {primary, backup}; objective = worst-case switch-controller propagation
latency over both levels (the latency after any single controller failure). Their exact instance data (loads, U) are
not available to us, so loads ~ U(100, 200) kreq/s (seeded) and U is set so that total load / (k U) = rho_c
(0.5 and 0.8). Propagation latency = shortest-path distance (km) / 2e5 km/s, from the cached Topology Zoo graphs.

Metaheuristic objective for a controller set C: binary search over the latencies D[:, C] for the smallest z at which
a greedy-regret assignment fits both levels within capacity (an upper bound on the exact z(C)); infeasible sets get
2 * max latency + overflow. Every reported value is then re-scored EXACTLY: z*(C) for the returned C by a MILP with
C fixed, and the global optimum z* by the full FFCCPP MILP (scipy HiGHS, time limit; the bound gap is reported when
not proved optimal).
Algorithms (v3 tuned settings, 20,000 evaluations, 10 runs each): random, ga, pso, sa, gwo, gwo_sa,
hybrid_eho_aco, hybrid_linear. Reported per instance: mean / best exact z of each method, gap to z*, wins."""

import json
import multiprocessing as mp
import time

import numpy as np
from numba import njit
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shield.cli import parser
from shield.placement.runner import run_algorithm, topology_and_latency, tuned_cfg
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, load_json, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("ffccpp")
ALGS = ["random", "ga", "pso", "sa", "gwo", "gwo_sa", "hybrid_eho_aco", "hybrid_linear"]
INSTANCES = [("Abilene", (2, 3, 4)), ("Geant2012", (3, 5, 7)), ("syn100", (4, 6)), ("Cogentco", (4, 6, 8)),
             ("Kdl", (6, 8))]
RHOS = (0.5, 0.8)
RUNS, BUDGET = 10, 20_000


@njit
def _fits2(Dsub, L, U, z, prim):
    # Within the latency limit z any controller is acceptable, so this is bin packing: most constrained switches
    # first (fewest controllers within z, then largest load), each to the in-range controller with most room.
    n, k = Dsub.shape
    opts = (Dsub <= z).sum(1)
    order = np.argsort(opts * 1e6 - L)
    for level in range(2):
        rem = np.full(k, U)
        for t in range(n):
            i = order[t]
            best, br = -1, -1.0
            for j in range(k):
                if level == 1 and j == prim[i]:
                    continue
                if Dsub[i, j] <= z and rem[j] >= L[i] and rem[j] > br:
                    best, br = j, rem[j]
            if best < 0:
                return False
            rem[best] -= L[i]
            if level == 0:
                prim[i] = best
    return True


@njit
def heuristic_z(Dsub, L, U, dmax):
    vals = np.unique(Dsub.ravel())
    prim = np.zeros(Dsub.shape[0], np.int64)
    if not _fits2(Dsub, L, U, vals[-1], prim):
        return 2.0 * dmax + L.sum()
    lo, hi = 0, len(vals) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _fits2(Dsub, L, U, vals[mid], prim):
            hi = mid
        else:
            lo = mid + 1
    return vals[lo]


class FFCCPP:
    """Duck-typed placement problem: the optimisers read D, lam, n, k and call problem(C)."""

    def __init__(self, D, L, k, U):
        self.D, self.lam, self.L = np.ascontiguousarray(D), np.ascontiguousarray(L), np.ascontiguousarray(L)
        self.n, self.k, self.U = len(D), k, float(U)
        self.dmax = float(D.max())

    def __call__(self, C):
        return float(heuristic_z(np.ascontiguousarray(self.D[:, np.asarray(C, np.int64)]), self.L, self.U, self.dmax))


def milp_ffccpp(D, L, k, U, fixed=None, time_limit=600):
    """Exact FFCCPP. Vars: y_j (n), x0_ij, x1_ij (n*m each, m = candidate sites), z. fixed = controller set."""
    n = len(D)
    sites = np.arange(n) if fixed is None else np.asarray(fixed)
    m = len(sites)
    ny, nx = (m if fixed is None else 0), n * m
    iz = ny + 2 * nx
    nv = iz + 1
    x0 = lambda i, j: ny + i * m + j
    x1 = lambda i, j: ny + nx + i * m + j
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(cs, vs, l, h):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs); lo.append(l); hi.append(h); r += 1

    for i in range(n):
        add([x0(i, j) for j in range(m)], [1] * m, 1, 1)
        add([x1(i, j) for j in range(m)], [1] * m, 1, 1)
        for j in range(m):
            add([x0(i, j), x1(i, j)] + ([j] if fixed is None else []), [1, 1] + ([-1] if fixed is None else []), -np.inf,
                0 if fixed is None else 1)
            add([x0(i, j), x1(i, j), iz], [D[i, sites[j]], D[i, sites[j]], -1], -np.inf, 0)
    for j in range(m):
        for lev in (x0, x1):
            add([lev(i, j) for i in range(n)] + ([j] if fixed is None else []), list(L) + ([-U] if fixed is None else []),
                -np.inf, 0 if fixed is None else U)
    if fixed is None:
        add(list(range(m)), [1] * m, k, k)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv))
    c = np.zeros(nv); c[iz] = 1
    integ = np.ones(nv); integ[iz] = 0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), np.r_[np.ones(nv - 1), np.inf]),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-6})
    if res.x is None:
        return None
    out = {"z": float(res.fun), "status": int(res.status), "bound": float(getattr(res, "mip_dual_bound", np.nan) or np.nan)}
    if fixed is None:
        out["C"] = np.flatnonzero(res.x[:m] > 0.5).tolist()
    return out


_W = {}


def _init():
    _W["pcfg"] = load_config("placement", False)
    _W["paths"] = Paths(False)
    tf = _W["paths"].v3 / "placement" / "tuned_params.json"
    _W["tuned"] = load_json(tf) if tf.exists() else None


def instance(name, k, rho, paths):
    topo, D = topology_and_latency(name, str(paths.topologies))
    L = np.random.default_rng(1000 + topo.n).uniform(100, 200, topo.n)
    U = L.sum() / (k * rho)
    return topo, D, L, U


def _run(task):
    name, k, rho, alg, seed = task
    topo, D, L, U = instance(name, k, rho, _W["paths"])
    prob = FFCCPP(D, L, k, U)
    cfg = tuned_cfg(_W["pcfg"], alg, _W["tuned"])
    t = time.time()
    res = run_algorithm(alg, prob, topo, cfg, seed, budget=BUDGET)
    ex = milp_ffccpp(D, L, k, U, fixed=res.best, time_limit=120)
    return {"topology": name, "n": int(topo.n), "k": k, "rho": rho, "alg": alg, "seed": seed, "C": res.best.tolist(),
            "heuristic_z": res.best_F, "exact_z": ex["z"] if ex else None, "seconds": time.time() - t}


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    P = Paths(False)
    od = P.v3 / "ffccpp"; od.mkdir(parents=True, exist_ok=True)
    runs_f, opt_f = od / "runs.jsonl", od / "optimum.json"
    done = {(r["topology"], r["k"], r["rho"], r["alg"], r["seed"]) for r in read_jsonl(runs_f)}
    tasks = [(t, k, rho, a, s) for t, ks in INSTANCES for k in ks for rho in RHOS for a in ALGS for s in range(RUNS)
             if (t, k, rho, a, s) not in done]
    log.info("%d runs to do", len(tasks))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for row in pool.imap_unordered(_run, tasks, chunksize=4):
            append_jsonl(row, runs_f)
    # global optima (small/medium instances; time-limited)
    opt = json.loads(opt_f.read_text()) if opt_f.exists() else {}
    for t, ks in INSTANCES:
        for k in ks:
            for rho in RHOS:
                key = f"{t}|{k}|{rho}"
                if key in opt:
                    continue
                topo, D, L, U = instance(t, k, rho, P)
                if topo.n > 200:
                    opt[key] = None; continue
                tt = time.time()
                r = milp_ffccpp(D, L, k, U, time_limit=1800)
                opt[key] = {**r, "seconds": time.time() - tt} if r else None
                log.info("optimum %s: %s", key, opt[key])
                save_json(opt, opt_f)
    save_json(opt, opt_f)


if __name__ == "__main__":
    main()
