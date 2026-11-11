"""P2 v4, X5 validation: proves or tightens the p_min values of scripts/79 (outputs_v5/p2/modelb_draws.jsonl).

scripts/79 takes p_min as the smallest p whose time-limited (30 min) ILP returned a feasible placement; a solve that
stopped without one moved the loop to p + 1, so p_min is an upper bound (one case is non-monotone in capacity). Here,
per row and per grid ratio that brackets a measured scenario (0.90, 0.80, 0.70, 0.60, 0.45, 0.40):
  UB = monotone envelope, min of p_min over all ratios at or below this one (less capacity never needs fewer);
  LB = max(K, ceil(K sum L / U)), the capacity bound;
  then a feasibility ILP (zero objective, same constraints as the FFCCP with closest assignment) at UB - 1, UB - 2,
  ... down to LB, 10 min each: infeasible proves UB; feasible lowers UB; a time limit leaves [LB, UB].
Output: outputs_v5/p2/modelb_close.jsonl (one line per row and ratio)."""
import importlib.util
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
RATIOS = (0.90, 0.80, 0.70, 0.60, 0.45, 0.40)
TL = 600
_W = {}


def _mods():
    if "m84" not in _W:
        from scipy.optimize import milp as _milp
        s = importlib.util.spec_from_file_location("m84", Path(__file__).with_name("84_p2v4_robust_placement.py"))
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)

        def feas(c, **kw):
            kw["options"] = {**kw.get("options", {}), "time_limit": TL}
            res = _milp(np.zeros_like(c), **kw); _W["status"] = int(res.status)
            return res
        m.milp = feas
        s2 = importlib.util.spec_from_file_location("kr57", Path(__file__).with_name("57_killi_rao.py"))
        k = importlib.util.module_from_spec(s2); s2.loader.exec_module(k)
        _W["m84"], _W["kr"] = m, k
    return _W["m84"], _W["kr"]


def task(t):
    row, r = t
    m84, kr = _mods()
    from shield.utils.config import Paths
    D, _ = kr.instance(row["topology"], Paths(False)); n = len(D); K = row["K"]
    L = np.random.default_rng(5000 + 100 * row["draw"] + n).uniform(100, 400, n); U = 7800.0 * r
    grid = sorted(((float(k[1:]), v["p_min"]) for k, v in row["scenarios"].items() if k.startswith("r")))
    ub = min(p for a, p in grid if a <= r + 1e-9 and p is not None)
    lb = max(K, math.ceil(K * L.sum() / U - 1e-9))
    m84.K = K
    steps, p, proven = [], ub - 1, ub == lb
    while p >= lb and not proven:
        _W.pop("status", None); res = m84.milp_cc(D, [L], [U], p, 0.0); st = _W.get("status")
        steps.append({"p": p, "status": st})
        if res is not None and st == 0:
            ub = p; p -= 1; proven = ub == lb
        elif st == 2:
            proven = True
        else:
            break
    return {"topology": row["topology"], "K": K, "draw": row["draw"], "ratio": r, "lb": lb, "ub": ub,
            "exact": bool(proven), "reported": dict(grid)[r], "steps": steps}


def main():
    rows = [json.loads(l) for l in open(OUT / "modelb_draws.jsonl")]
    out_f = OUT / "modelb_close.jsonl"
    done = set()
    if out_f.exists():
        done = {(x["topology"], x["K"], x["draw"], x["ratio"]) for x in map(json.loads, open(out_f))}
    tasks = [(row, r) for row in rows for r in RATIOS if (row["topology"], row["K"], row["draw"], r) not in done]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(len(tasks), "tasks", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, open(out_f, "a") as f:
        for x in pool.imap_unordered(task, tasks):
            f.write(json.dumps(x) + "\n"); f.flush()
            print(x["topology"], x["K"], x["draw"], x["ratio"], x["lb"], x["ub"], x["exact"], x["reported"], flush=True)


if __name__ == "__main__":
    main()
