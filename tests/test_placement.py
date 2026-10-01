import networkx as nx
import numpy as np
import pytest

from shield.eval.resilience import controller_failures, network_failures
from shield.eval.stats import holm
from shield.placement.common import Evaluator, mix
from shield.placement.exact import brute_force
from shield.placement.objective import Instance, PlacementProblem, assign_capacitated
from shield.placement.runner import STOCHASTIC, run_algorithm
from shield.placement.topology import Topology, latency_matrix, synthetic_iot

W = {"avg": 1.0, "max": 0.5, "inter": 0.1, "imbalance": 0.2, "fail": 1.0}


def path_graph(n=5):
    G = nx.path_graph(n)
    nx.set_edge_attributes(G, 1.0, "delay")
    return G


def test_objective_on_path_graph():
    G = path_graph()
    D = latency_matrix(G)
    inst = Instance("path", D, np.ones(5), mu=1e9, k=1)
    P = PlacementProblem(inst, W, 0.95, 10.0, sla_ms=1.5, n_reference=5)
    d = P.describe(np.array([2]))
    assert d["avg"] == pytest.approx(1.2, abs=1e-5)  # (2+1+0+1+2)/5, queueing negligible at mu=1e9
    assert d["max"] == pytest.approx(2.0, abs=1e-5)
    assert d["pct_sla"] == pytest.approx(0.6)


def test_failover_term_k2():
    D = latency_matrix(path_graph())
    P = PlacementProblem(Instance("path", D, np.ones(5), 1e9, 2), W, 0.95, 10.0, 10.0, n_reference=5)
    d = P.describe(np.array([0, 4]))
    # Losing controller 0 sends switches 0,1,2 to node 4: latencies 4,3,2,1,0 -> mean 2.
    assert d["fail"] == pytest.approx(2.0, abs=1e-5)


def test_assignment_respects_capacity():
    Dsub = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [5.0, 1.0]])
    lam = np.array([1.0, 1.0, 1.0, 1.0])
    primary, backup = assign_capacitated(Dsub, lam, 2.0)
    loads = np.bincount(primary, weights=lam, minlength=2)
    assert (loads <= 2.0).all()
    assert (primary != backup).all()


def test_mix_keeps_k_distinct():
    rng = np.random.default_rng(0)
    for _ in range(200):
        base, guide = np.sort(rng.choice(30, 5, replace=False)), np.sort(rng.choice(30, 5, replace=False))
        out = mix(base, guide, 0.7, rng)
        assert len(out) == 5 and len(set(out.tolist())) == 5


def _small_problem(n=25, k=3, rho=0.7):
    topo = synthetic_iot(n, seed=3)
    D = latency_matrix(topo.G)
    mu = 1000.0
    lam = np.random.default_rng(0).lognormal(0, 1, n)
    lam = lam / lam.sum() * rho * k * mu
    return topo, PlacementProblem(Instance(topo.name, D, lam, mu, k, topo.coords), W, 0.95, 10.0, 20.0, 50)


def test_hybrid_finds_brute_force_optimum():
    topo, P = _small_problem()
    opt = brute_force(P, 10**6)
    cfg = {"budget": 1500,
           "hybrid": {"n_clans": 5, "clan_size": 8, "alpha": 0.5, "beta": 0.3, "ants_per_clan": 1,
                      "local_search_every": 10, "local_search_evals": 40},
           "aco": {"n_ants": 20, "a": 1.0, "b": 2.0, "evaporation": 0.1, "mask_quantile": 0.05}}
    hits = sum(abs(run_algorithm("hybrid_eho_aco", P, topo, cfg, s).best_F - opt["best_F"]) < 1e-9 for s in range(30))
    assert hits >= 29  # >= 95% of 30 seeds


@pytest.mark.parametrize("alg", STOCHASTIC + ["kmedian", "kcenter", "pagerank", "ilp_ckm"])
def test_every_algorithm_returns_valid_solution(alg):
    topo, P = _small_problem(n=20, k=3)
    cfg = {"budget": 300, "ilp_time_limit": 20,
           "hybrid": {"n_clans": 3, "clan_size": 5, "alpha": 0.5, "beta": 0.3, "ants_per_clan": 1,
                      "local_search_every": 5, "local_search_evals": 20},
           "aco": {"n_ants": 10, "a": 1.0, "b": 2.0, "evaporation": 0.1, "mask_quantile": 0.05},
           "ga": {"pop": 20, "cx_rate": 0.9, "mut_rate": 0.2, "elite": 2, "tournament": 3},
           "pso": {"swarm": 15, "w": 0.4, "c1": 0.3, "c2": 0.3}, "sa": {"t0_samples": 20, "cooling": 0.99}}
    res = run_algorithm(alg, P, topo, cfg, seed=1)
    assert res is not None and len(set(res.best.tolist())) == 3 and res.n_evals <= 300
    assert res.best_F == pytest.approx(P(res.best))


def test_evaluator_budget():
    _, P = _small_problem(n=15, k=2)
    ev = Evaluator(P, 5)
    for _ in range(5):
        ev(np.array([0, 1]))
    assert ev.exhausted


def test_resilience_reports():
    G = path_graph(6)
    D = latency_matrix(G)
    P = PlacementProblem(Instance("p", D, np.ones(6), 1e9, 2), W, 0.95, 10.0, 3.0, n_reference=5)
    C = np.array([1, 4])
    rep = controller_failures(P, C, 10, np.random.default_rng(0))
    assert rep["single"]["worst_pct_orphan"] == 0.0
    # Cutting the middle link orphans nobody (each half keeps a controller) ...
    r = network_failures(G, P, C, "link", 1 / 5, True, 1, np.random.default_rng(0), {(2, 3): 1.0})
    assert r["mean_pct_orphan"] == 0.0
    # ... and topology wrapper sanity
    assert Topology("p", G, np.zeros((6, 2))).n == 6


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
