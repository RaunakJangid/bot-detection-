"""Resilience of a placement under controller, link and node failures.

Failover rule everywhere: a switch keeps its primary if it is alive and reachable, else its backup,
else the nearest reachable surviving controller; with none reachable it is orphaned.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from shield.placement.objective import PlacementProblem, assign_capacitated


def _failover(T: np.ndarray, primary: np.ndarray, backup: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """T: (n, k) latency to each controller (inf if unreachable). Returns assignment (-1 = orphan)."""
    n = len(T)
    reach = np.isfinite(T) & alive[None, :]
    rows = np.arange(n)
    assign = np.where(reach[rows, primary], primary, -1)
    use_b = (assign < 0) & reach[rows, backup]
    assign[use_b] = backup[use_b]
    rest = (assign < 0) & reach.any(1)
    if rest.any():
        masked = np.where(reach[rest], T[rest], np.inf)
        assign[rest] = masked.argmin(1)
    return assign


def _metrics(T: np.ndarray, assign: np.ndarray, lam: np.ndarray, mu: float, cap: float, sla: float,
             active: np.ndarray, alive: np.ndarray) -> dict:
    """active: switches that still exist; alive: controllers that still exist."""
    k = T.shape[1]
    ok = active & (assign >= 0)
    loads = np.bincount(assign[ok], weights=lam[ok], minlength=k)
    wait = 1000.0 / (mu - np.minimum(loads, cap))
    lat = np.full(len(T), np.inf)
    lat[ok] = T[ok, assign[ok]] + wait[assign[ok]]
    n_active = max(int(active.sum()), 1)
    finite = lat[ok]
    return {
        "avg_ms": float(finite.mean()) if len(finite) else float("nan"),
        "max_ms": float(finite.max()) if len(finite) else float("nan"),
        "pct_sla": float((lat[active] <= sla).sum() / n_active),
        "pct_orphan": float((active & (assign < 0)).sum() / n_active),
        "overload_ratio": float((loads[alive] > cap).mean()) if alive.any() else 1.0,
    }


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        out[f"mean_{key}"] = float(vals.mean())
        worse_is_low = key in ("pct_sla",)
        out[f"worst_{key}"] = float(vals.min() if worse_is_low else vals.max())
    return out


def controller_failures(problem: PlacementProblem, C: np.ndarray, max_double: int,
                        rng: np.random.Generator) -> dict:
    T = problem.D[:, C]
    primary, backup = assign_capacitated(np.ascontiguousarray(T), problem.lam, problem.cap)
    k = len(C)
    active = np.ones(problem.n, dtype=bool)
    out = {}
    combos = {"single": [(f,) for f in range(k)], "double": list(itertools.combinations(range(k), 2))}
    if len(combos["double"]) > max_double:
        pick = rng.choice(len(combos["double"]), max_double, replace=False)
        combos["double"] = [combos["double"][i] for i in pick]
    for kind, scenarios in combos.items():
        rows = []
        for failed in scenarios:
            alive = np.ones(k, dtype=bool)
            alive[list(failed)] = False
            if not alive.any():
                continue
            rows.append(_metrics(T, _failover(T, primary, backup, alive), problem.lam, problem.mu, problem.cap,
                                 problem.sla, active, alive))
        if rows:
            out[kind] = _aggregate(rows)
    return out


def _latency_from(G: nx.Graph, n: int, sources: np.ndarray, dead_nodes: set[int], dead_edges: set) -> np.ndarray:
    rows, cols, w = [], [], []
    for u, v, d in G.edges(data="delay"):
        if u in dead_nodes or v in dead_nodes or (u, v) in dead_edges or (v, u) in dead_edges:
            continue
        rows += [u, v]; cols += [v, u]; w += [d, d]
    A = csr_matrix((w, (rows, cols)), shape=(n, n))
    return dijkstra(A, directed=False, indices=sources).T  # (n, k)


def network_failures(G: nx.Graph, problem: PlacementProblem, C: np.ndarray, kind: str, frac: float,
                     targeted: bool, trials: int, rng: np.random.Generator,
                     betweenness: dict | None = None) -> dict:
    """kind: 'link' or 'node' (controller nodes are never removed here; see controller_failures)."""
    n = problem.n
    T0 = problem.D[:, C]
    primary, backup = assign_capacitated(np.ascontiguousarray(T0), problem.lam, problem.cap)
    alive = np.ones(len(C), dtype=bool)
    ctrl = set(C.tolist())
    rows = []
    for _ in range(1 if targeted else trials):
        dead_nodes, dead_edges = set(), set()
        if kind == "link":
            edges = list(G.edges)
            m = max(1, int(round(frac * len(edges))))
            if targeted:
                ranked = sorted(edges, key=lambda e: betweenness.get(e, betweenness.get((e[1], e[0]), 0.0)),
                                reverse=True)
                dead_edges = set(ranked[:m])
            else:
                dead_edges = {edges[i] for i in rng.choice(len(edges), m, replace=False)}
        else:
            cands = [v for v in range(n) if v not in ctrl]
            m = max(1, int(round(frac * n)))
            if targeted:
                dead_nodes = set(sorted(cands, key=lambda v: betweenness.get(v, 0.0), reverse=True)[:m])
            else:
                dead_nodes = {cands[i] for i in rng.choice(len(cands), min(m, len(cands)), replace=False)}
        T = _latency_from(G, n, C, dead_nodes, dead_edges)
        active = np.ones(n, dtype=bool)
        active[list(dead_nodes)] = False
        rows.append(_metrics(T, _failover(T, primary, backup, alive), problem.lam, problem.mu, problem.cap,
                             problem.sla, active, alive))
    return _aggregate(rows)


def resilience_report(G: nx.Graph, problem: PlacementProblem, C: np.ndarray, cfg: dict, seed: int,
                      edge_bc: dict, node_bc: dict) -> dict:
    rng = np.random.default_rng(seed)
    rep = {"controller": controller_failures(problem, C, cfg["max_double"], rng)}
    for kind, fracs, bc in (("link", cfg["link_fail_fracs"], edge_bc), ("node", cfg["node_fail_fracs"], node_bc)):
        for frac in fracs:
            for targeted in (False, True):
                key = f"{kind}_{'targeted' if targeted else 'random'}_{int(frac * 100)}"
                rep[key] = network_failures(G, problem, C, kind, frac, targeted, cfg["trials"], rng, bc)
    return rep


def betweenness(G: nx.Graph) -> tuple[dict, dict]:
    k = None if G.number_of_nodes() <= 300 else 200  # sampled estimate on large graphs
    return (nx.edge_betweenness_centrality(G, k=k, weight="delay", seed=0),
            nx.betweenness_centrality(G, k=k, weight="delay", seed=0))
