"""P2 v4: EHO-ACO component ablation in the current setting (re-run of the v2 Model A ablation,
outputs_v2/paper/tables/placement_ablation.csv, which used the earlier EHO-ACO version, Model A and the full budget only).

Same benchmark and protocol as X8 (scripts/82): the 55 FFCCP instances, 30 seeds x 20,000 evaluations, best value after
1k, 2k, 5k, 10k and 20k evaluations. Every variant uses the identical Algorithm 1 setting (tuned hybrid_eho_aco parameters
for the shared hybrid.* and aco.* sections); only the removed component differs:
  hybrid_eho_aco  full method (= X8 'hybrid_eho_aco:tuned', re-run here for a matched comparison)
  hybrid_no_ants  EHO clans + SA intensification; worst elephants rebuilt at random instead of by ACO ants
  hybrid_no_ls    EHO clans + ACO ants; no SA intensification
  eho             EHO alone (random restarts, no ants, no SA)
  aco             MAX-MIN ACO alone
  aco_sa          MAX-MIN ACO + the same SA intensification (every local_search_every colony iterations, same
                  evaluations and start temperature), the improved solution reinforcing the pheromone (added after
                  review: separates the hybridisation effect from the SA effect)
SA (X8 'sa:cool999') is the external reference. Analysis (--analyse): mean ratio z/z_ref (infeasible = 2) per budget,
Friedman over instances, Wilcoxon of the full method against each variant with Holm correction, on all 55 and on the 12
large instances. Output: outputs_v5/p2/ablation_runs.jsonl, ablation_stats.json."""
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

OUT = Path("D:/shield_run/outputs_v5/p2")
VARIANTS = ("hybrid_eho_aco", "hybrid_no_ants", "hybrid_no_ls", "eho", "aco", "aco_sa")
SEEDS, CHECK = 30, (250, 500, 1000, 2000, 5000, 10000, 20000)


def run_variant(alg, prob, topo, cfg, seed, budget):
    """All variants through one entry; every objective call is counted by the Evaluator (cache hits included)."""
    import numpy as np
    from shield.placement.aco import Colony
    from shield.placement.common import BudgetExhausted, Evaluator, nearest_lists
    from shield.placement.hybrid_eho_aco import sa_intensify
    from shield.placement.runner import run_algorithm
    if alg != "aco_sa":
        return run_algorithm(alg, prob, topo, cfg, seed, budget=budget)
    rng = np.random.default_rng(seed); ev = Evaluator(prob, budget); k = prob.k; h, a = cfg["hybrid"], cfg["aco"]
    colony = Colony(prob.D, prob.lam, a, rng); near = nearest_lists(prob.D); it = 0
    try:
        while not ev.exhausted:
            scored = [(C, ev(C)) for C in (colony.construct(k) for _ in range(min(a["n_ants"], ev.remaining)))]
            colony.update([min(scored, key=lambda t: t[1]), (ev.best, ev.best_F)], ev.best_F); it += 1
            if it % h["local_search_every"] == 0:
                b, bF = sa_intensify(ev.best, ev.best_F, ev, near, h["local_search_evals"], h["sa_t0_frac"], rng)
                colony.update([(b, bF)], ev.best_F)
    except BudgetExhausted:
        pass
    return ev.result() if hasattr(ev, "result") else type("R", (), {"best_F": ev.best_F, "best": ev.best, "history": ev.history})()
_M = {}


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _init():
    _M["m63"] = _load("63_placement_cv.py"); _M["m63"]._init()


def _run(task):
    from shield.placement.runner import tuned_cfg
    m63 = _M["m63"]; kr = m63.kr
    name, K, p, alg, seed = task
    D, L = kr.instance(name, kr._W["paths"])
    prob = kr.FFCCP(D, L, p, K)
    cfg = tuned_cfg(kr._W["pcfg"], "hybrid_eho_aco", kr._W["tuned"])          # Algorithm 1 setting for every variant
    res = run_variant(alg, prob, kr._Topo(), cfg, seed, kr.BUDGET)
    hist = getattr(res, "history", None) or []
    any_ = {str(c): (float(min(f for e, f in hist if e <= c)) if any(e <= c for e, f in hist) else None) for c in CHECK}
    return {"topology": name, "K": K, "p": p, "alg": alg, "seed": seed, "z": float(res.best_F), "anytime": any_}


def run(workers):
    _init()
    from shield.utils.config import Paths
    small, large = _M["m63"].instances(Paths(False))
    out_f = OUT / "ablation_runs.jsonl"; done = set()
    if out_f.exists():
        done = {(r["topology"], r["K"], r["p"], r["alg"], r["seed"]) for r in map(json.loads, open(out_f))}
    tasks = [(t, K, p, a, s) for t, K, p in small + large for a in VARIANTS for s in range(SEEDS) if (t, K, p, a, s) not in done]
    print(len(tasks), "runs to do", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(out_f, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_run, tasks, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                f.flush(); print(i, r["topology"], r["p"], r["alg"], r["z"], flush=True)


def analyse():
    import numpy as np
    from scipy.stats import friedmanchisquare, wilcoxon
    R = [json.loads(l) for l in open(OUT / "ablation_runs.jsonl")]
    X8 = [json.loads(l) for l in open(OUT / "search_runs.jsonl")]
    R += [{**r, "alg": "sa"} for r in X8 if r["alg"] == "sa" and r["cand"] == "cool999"]
    opt = json.load(open("D:/shield_run/outputs_v3/killi_rao/optimum.json"))
    key = lambda i: f"{i[0]}|{i[1]}|{i[2]}"
    inst = sorted({(r["topology"], r["K"], r["p"]) for r in R})
    small = {i for i in inst if key(i) in opt and opt[key(i)]["status"] == 0}
    large = [i for i in inst if i not in small and i[0] in ("Cogentco", "Kdl")]
    best = defaultdict(lambda: np.inf)
    for r in R + X8:
        if r["z"] < 1000:
            best[(r["topology"], r["K"], r["p"])] = min(best[(r["topology"], r["K"], r["p"])], r["z"])
    zref = {i: (opt[key(i)]["z"] if i in small else best[i]) for i in inst}
    algs = list(VARIANTS) + ["sa"]; out = {}
    for b in [None] + list(CHECK[:-1]):
        acc = defaultdict(list)
        for r in R:
            i = (r["topology"], r["K"], r["p"])
            z = r["z"] if b is None else (r.get("anytime") or {}).get(str(b))
            if z is not None:
                acc[(i, r["alg"])].append(2.0 if z >= 1000 else z / zref[i])
        lab = "full" if b is None else str(b); out[lab] = {}
        for subset_name, subset in (("all", inst), ("large", large)):
            A = [a for a in algs if all(acc[(i, a)] for i in subset)]
            M = np.array([[np.mean(acc[(i, a)]) for a in A] for i in subset])
            res = {"mean_ratio": {a: float(M[:, j].mean()) for j, a in enumerate(A)}}
            if len(A) >= 3:
                res["friedman_p"] = float(friedmanchisquare(*M.T).pvalue)
            pw = []
            for j, a in enumerate(A):
                if a == "hybrid_eho_aco":
                    continue
                x, y = M[:, A.index("hybrid_eho_aco")], M[:, j]; d = x - y
                p = float(wilcoxon(x, y).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
                pw.append([a, int((d < -1e-12).sum()), int((d > 1e-12).sum()), p])
            order = np.argsort([x[3] for x in pw]); adj = [0.0] * len(pw); run_ = 0.0
            for kk, idx in enumerate(order):
                run_ = max(run_, min(1.0, (len(pw) - kk) * pw[idx][3])); adj[idx] = run_
            res["full_vs"] = {x[0]: {"full_better": x[1], "full_worse": x[2], "p": x[3], "p_holm": adj[n]} for n, x in enumerate(pw)}
            out[lab][subset_name] = res
    json.dump(out, open(OUT / "ablation_stats.json", "w"), indent=1)
    for lab, v in out.items():
        for sub, r in v.items():
            print(lab, sub, {a: round(x, 3) for a, x in r["mean_ratio"].items()})
            for a, t in r["full_vs"].items():
                print("    full vs", a, t)


if __name__ == "__main__":
    analyse() if "--analyse" in sys.argv else run(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
