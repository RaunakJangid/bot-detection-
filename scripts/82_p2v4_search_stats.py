"""P2 v4, experiment X8: search methods on the published FFCCP model with 30 seeds and proper statistics. CPU only.

Instances: the 43 Killi & Rao instances with proven optima and the 12 large Cogentco / Kdl settings (scripts/63).
Methods with the setting chosen most often by the 5-fold cross-validation of scripts/63 (outputs_v3/placement_cv):
EHO-ACO "ls_every5", GWO+SA "tuned", PSO-SA "default", SA "cool999", GA "tuned"; plus EHO-ACO "tuned" (Algorithm 1 of
v3) for reference. 30 seeds x 20,000 evaluations per instance and method; the best value is also recorded after
1,000, 2,000, 5,000, 10,000 and 20,000 evaluations where the method exposes its history (anytime quality).
Output: outputs_v5/p2/search_runs.jsonl; analysis in scripts/83."""

import importlib.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

OUT = Path("D:/shield_run/outputs_v5/p2"); OUT.mkdir(parents=True, exist_ok=True)
METHODS = [("hybrid_eho_aco", "ls_every5"), ("hybrid_eho_aco", "tuned"), ("gwo_sa", "tuned"), ("pso_sa", "default"),
           ("sa", "cool999"), ("ga", "tuned")]
SEEDS, CHECK = 30, (1000, 2000, 5000, 10000, 20000)
_M = {}


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _init():
    _M["m63"] = _load("63_placement_cv.py")
    _M["m63"]._init()


def _run(task):
    from shield.placement.runner import run_algorithm
    m63 = _M["m63"]; kr, ext = m63.kr, m63.ext
    name, K, p, alg, cand, seed = task
    D, L = kr.instance(name, kr._W["paths"])
    prob = kr.FFCCP(D, L, p, K)
    cfg = m63.cfg_for(alg, cand)
    any_ = None
    if alg == "pso_sa":
        _, F, _ = ext.run_pso_sa(prob, cfg, seed, kr.BUDGET)
    else:
        res = run_algorithm(alg, prob, kr._Topo(), cfg, seed, budget=kr.BUDGET)
        F = res.best_F
        hist = getattr(res, "history", None)
        if hist:
            any_ = {}
            for c in CHECK:
                vals = [f for e, f in hist if e <= c]
                any_[str(c)] = float(min(vals)) if vals else None
    return {"topology": name, "K": K, "p": p, "alg": alg, "cand": cand, "seed": seed, "z": float(F), "anytime": any_}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    _init()
    from shield.utils.config import Paths
    small, large = _M["m63"].instances(Paths(False))
    out_f = OUT / "search_runs.jsonl"
    done = set()
    if out_f.exists():
        for ln in open(out_f):
            r = json.loads(ln); done.add((r["topology"], r["K"], r["p"], r["alg"], r["cand"], r["seed"]))
    tasks = [(t, K, p, a, c, s) for t, K, p in small + large for a, c in METHODS for s in range(SEEDS)
             if (t, K, p, a, c, s) not in done]
    print(f"{len(small)} small + {len(large)} large instances, {len(tasks)} runs to do", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(out_f, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_run, tasks, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                f.flush(); print(i, r["topology"], r["p"], r["alg"], r["z"], flush=True)


if __name__ == "__main__":
    main()
