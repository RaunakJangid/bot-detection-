"""Where does the hybrid's premature convergence come from? Per generation: pheromone entropy (normalised),
clan diversity (distinct nodes used by all elephants / n), and the best F. Cogentco k=8, tuned settings."""
import json

import numpy as np

from shield.placement.aco import Colony
from shield.placement.common import BudgetExhausted, Evaluator, nearest_lists
from shield.placement.eho import eho_generation, init_clans
from shield.placement.hybrid_eho_aco import sa_intensify
from shield.placement.runner import build_problem, controller_mu, topology_and_latency, tuned_cfg
from shield.utils.config import Paths, load_config

pcfg = load_config("placement")
paths = Paths()
cfg = tuned_cfg(pcfg, "hybrid_eho_aco", json.load(open(paths.v3 / "placement" / "tuned_params.json")))
mu = controller_mu(pcfg, 2853954)
for topo_name, k in (("Cogentco", 8), ("syn200", 6)):
    topo, D = topology_and_latency(topo_name, str(paths.topologies))
    P = build_problem(topo, D, k, 0.5 * k * mu, mu, cfg)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        ev = Evaluator(P, cfg["budget"])
        h, a = cfg["hybrid"], cfg["aco"]
        col, near = Colony(P.D, P.lam, a, rng), nearest_lists(P.D)
        log, gen = [], 0
        try:
            clans = init_clans(ev, k, h["n_clans"], h["clan_size"], rng, lambda: col.construct(k))
            while True:
                eho_generation(ev, clans, k, h, rng, lambda: col.construct(k))
                gen += 1
                col.update([min(c, key=lambda e: e[1]) for c in clans] + [(ev.best, ev.best_F)], ev.best_F)
                if gen % h["local_search_every"] == 0:
                    b, bf = sa_intensify(ev.best, ev.best_F, ev, near, h["local_search_evals"], h["sa_t0_frac"], rng)
                    wc = max(range(len(clans)), key=lambda i: min(e[1] for e in clans[i]))
                    clans[wc].sort(key=lambda e: e[1]); clans[wc][-1] = (b, bf)
                p = col.tau / col.tau.sum()
                ent = float(-(p * np.log(p)).sum() / np.log(len(p)))
                div = len(set(np.concatenate([C for c in clans for C, _ in c]).tolist())) / P.n
                log.append((ev.n_evals, ent, div, ev.best_F))
        except BudgetExhausted:
            pass
        L = np.array(log)
        q = [np.searchsorted(L[:, 0], f * cfg["budget"]) for f in (0.05, 0.2, 0.5, 0.99)]
        print(topo_name, k, seed, "| entropy", np.round(L[q, 1], 3), "| clan diversity", np.round(L[q, 2], 3),
              "| best F", np.round(L[q, 3], 4))
