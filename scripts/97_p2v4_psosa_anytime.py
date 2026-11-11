"""P2 v4, closes X8 gap O5: PSO-SA over-time records on the 12 large FFCCP instances.

scripts/82 ran PSO-SA through scripts/58 run_pso_sa, which returns only the final best, so X8 has no anytime values for
PSO-SA. This re-runs exactly the same runs (same instances, same 'default' setting, same 30 seeds, same 20,000-evaluation
budget, same deterministic generator) and keeps the evaluator's improvement history. A consistency check compares each
re-run's final value with the X8 value. Output: outputs_v5/p2/psosa_anytime_runs.jsonl."""
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
CHECK = (1000, 2000, 5000, 10000, 20000)
_M = {}


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _init():
    _M["m63"] = _load("63_placement_cv.py"); _M["m63"]._init()


def _run(task):
    from shield.placement.common import BudgetExhausted, Evaluator
    m63 = _M["m63"]; kr, ext = m63.kr, m63.ext
    name, K, p, seed = task
    D, L = kr.instance(name, kr._W["paths"])
    prob = kr.FFCCP(D, L, p, K)
    cfg = m63.cfg_for("pso_sa", "default")
    rng = np.random.default_rng(seed); ev = Evaluator(prob, kr.BUDGET)
    ext.run_pso(ext._Limited(ev, int(0.7 * kr.BUDGET)), prob.k, cfg["pso"], rng)
    try:
        ext.sa_intensify(ev.best, ev.best_F, ev, ext.nearest_lists(prob.D), ev.remaining, cfg["gwo_sa"]["t0_frac"], rng)
    except BudgetExhausted:
        pass
    any_ = {str(c): (float(min(f for e, f in ev.history if e <= c)) if any(e <= c for e, f in ev.history) else None)
            for c in CHECK}
    return {"topology": name, "K": K, "p": p, "alg": "pso_sa", "cand": "default", "seed": seed, "z": float(ev.best_F),
            "anytime": any_}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    x8 = {(r["topology"], r["K"], r["p"], r["seed"]): r["z"] for r in map(json.loads, open(OUT / "search_runs.jsonl"))
          if r["alg"] == "pso_sa"}
    _init()
    from shield.utils.config import Paths
    _, large = _M["m63"].instances(Paths(False))
    tasks = [(t, K, p, s) for t, K, p in large for s in range(30)]
    out_f = OUT / "psosa_anytime_runs.jsonl"
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(out_f, "w") as f:
        for r in pool.imap_unordered(_run, tasks):
            r["x8_z"] = x8.get((r["topology"], r["K"], r["p"], r["seed"]))
            r["matches_x8"] = r["x8_z"] is not None and abs(r["z"] - r["x8_z"]) <= 1e-9 * max(1.0, abs(r["x8_z"]))
            f.write(json.dumps(r) + "\n"); f.flush()
    print("done")


if __name__ == "__main__":
    main()
