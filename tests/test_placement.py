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
    # Within 0.1% of the true optimum on >= 95% of seeds. (Exact-hit counts vary across CPUs: tiny float
    # differences change tie-breaks; one machine hit the exact optimum 28/30 times, missing by 0.009%.)
    gaps = [(run_algorithm("hybrid_eho_aco", P, topo, cfg, s).best_F - opt["best_F"]) / opt["best_F"]
            for s in range(30)]
    assert min(gaps) >= -1e-12 and sum(g <= 1e-3 for g in gaps) >= 29


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


def test_zoo_download_falls_back_and_validates(tmp_path, monkeypatch):
    import shield.placement.topology as T
    # Three "mirrors": the first has no file, the second a drawing-only file (pixel x/y, no
    # Latitude/Longitude), the third the real file. Only the third may be accepted.
    dead, drawing, good = (tmp_path / d for d in ("dead", "drawing", "good"))
    for d in (dead, drawing, good):
        d.mkdir()
    H = nx.Graph()
    H.add_node("a", x=1.0, y=2.0)
    nx.write_graphml(H, drawing / "Net.graphml")
    G = nx.Graph()
    G.add_node("a", Latitude=40.0, Longitude=-74.0)
    G.add_node("b", Latitude=41.9, Longitude=-87.6)
    G.add_edge("a", "b")
    nx.write_graphml(G, good / "Net.graphml")
    monkeypatch.setattr(T, "ZOO_URLS", tuple(d.as_uri() + "/{name}.graphml" for d in (dead, drawing, good)))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "Net.graphml").write_text("<graphml truncated")  # broken cached copy must be replaced
    topo = T.load_zoo("Net", cache)
    assert topo.n == 2 and not list(cache.glob("*.part"))
    assert topo.G.edges[0, 1]["delay"] == pytest.approx(T.haversine_km(40.0, -74.0, 41.9, -87.6) / T.KM_PER_MS)

    monkeypatch.setattr(T, "ZOO_URLS", tuple(d.as_uri() + "/{name}.graphml" for d in (dead, drawing)))
    with pytest.raises(RuntimeError, match="not a valid graphml"):
        T.download_zoo("Net", tmp_path / "cache2")


def test_with_params_and_tuned_cfg():
    from shield.placement.runner import tuned_cfg, with_params
    cfg = {"hybrid": {"n_clans": 5, "alpha": 0.5}, "aco": {"b": 2.0}, "budget": 10}
    out = with_params(cfg, {"hybrid.n_clans": 8, "aco.b": 4.0})
    assert out["hybrid"] == {"n_clans": 8, "alpha": 0.5} and out["aco"]["b"] == 4.0
    assert cfg["hybrid"]["n_clans"] == 5  # original untouched
    tuned = {"sa": {"params": {"sa.cooling": 0.99}}}
    assert tuned_cfg({"sa": {"cooling": 0.999}}, "sa", tuned)["sa"]["cooling"] == 0.99
    assert tuned_cfg(cfg, "ga", tuned) is cfg


def test_tuning_candidates_score_choose():
    from shield.placement.tuning import candidates, choose, score, tuning_instances
    rng = np.random.default_rng(0)
    cands = candidates({"a.x": [1, 2, 3], "a.y": [0.1, 0.2]}, 4, rng)
    assert cands[0] == {} and len(cands) == 4 and len({tuple(sorted(c.items())) for c in cands}) == 4
    rows = [{"algorithm": "ga", "candidate": c, "topology": "t", "k": 2, "run": 0, "F": f}
            for c, f in ((0, 2.0), (1, 1.0), (2, 1.5))]
    rows += [{"algorithm": "sa", "candidate": 0, "topology": "t", "k": 2, "run": 0, "F": 1.2}]
    s = score(rows)
    tuned = choose(s, {"ga": [{}, {"ga.pop": 20}, {"ga.pop": 80}], "sa": [{}]}, {"ga_clone": "ga"})
    assert tuned["ga"]["params"] == {"ga.pop": 20} and tuned["ga"]["score"] == pytest.approx(1.0)
    assert tuned["ga"]["default_score"] == pytest.approx(2.0)
    assert tuned["ga_clone"]["inherited_from"] == "ga"
    insts = tuning_instances({"synthetic": [50], "k_values": [4], "seed_offset": 1000})
    assert insts == [("syn50s1050", 4)]


def test_tuning_graphs_differ_from_evaluation_graphs():
    from shield.placement.topology import get_topology
    a, b = get_topology("syn30", None), get_topology("syn30s1030", None)
    assert a.n == b.n == 30 and b.name == "syn30s1030"
    assert not np.allclose(a.coords, b.coords)


def test_sa_intensify_never_worsens():
    from shield.placement.common import nearest_lists
    from shield.placement.hybrid_eho_aco import sa_intensify
    topo, P = _small_problem(n=25, k=3)
    ev = Evaluator(P, 500)
    C0 = np.array([0, 1, 2])
    F0 = ev(C0)
    C, F = sa_intensify(C0, F0, ev, nearest_lists(P.D), 200, 0.05, np.random.default_rng(0))
    assert F <= F0 and F == pytest.approx(P(C)) and ev.n_evals <= 201


def test_relative_sla():
    from shield.placement.coupling import topology_sla
    D = latency_matrix(path_graph(6))
    lam = np.ones(6)
    assert topology_sla(D, lam, {"mode": "absolute"}, 2, 20.0) == 20.0
    from shield.placement.baselines import greedy_kmedian
    sla = topology_sla(D, lam, {"mode": "relative", "quantile": 1.0, "slack": 1.5}, 2, 20.0)
    # greedy 2-median on a 6-node path picks the middle node first: worst switch is 2 hops away
    worst = D[:, greedy_kmedian(D, lam, 2)].min(1).max()
    assert worst == pytest.approx(2.0) and sla == pytest.approx(1.5 * worst)


def test_wilcoxon_reports_both_directions():
    import pandas as pd
    from shield.eval.stats import wilcoxon_vs
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(20):
        base = rng.normal(10, 0.1)
        for alg, f in (("ours", base), ("worse_alg", base + 1.0), ("better_alg", base - 1.0)):
            rows.append({"topology": "t", "k": 2, "rho": 0.5, "algorithm": alg, "seed": seed, "F": f})
    w = wilcoxon_vs(pd.DataFrame(rows), "ours", ["worse_alg", "better_alg"]).set_index("vs")
    assert w.loc["worse_alg", "p_holm"] < 0.05 and w.loc["worse_alg", "p_worse_holm"] > 0.5
    assert w.loc["better_alg", "p_worse_holm"] < 0.05 and w.loc["better_alg", "p_holm"] > 0.5


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
