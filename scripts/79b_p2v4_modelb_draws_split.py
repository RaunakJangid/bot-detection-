"""P2 v4, X5 completion runner: computes the rows of scripts/79 that are still missing, with the work split per
(row, capacity scenario) instead of per row, so more cores can be used and every finished piece is saved.

Identical computation to scripts/79 `_task` (same instance, loads, planned p0 via pmin at U0, same pmin search from the
same capacity lower bound, same 1,800 s ILP time limit, same output row format):
  phase 1: per missing (topology, K, draw): p0 and planned set C0 (checkpoint modelb_split_plan.jsonl);
  phase 2: per (row, ratio): planned utilisation; if > 1, pmin at U0 * ratio (checkpoint modelb_split_parts.jsonl);
  assembly: complete rows appended to modelb_draws.jsonl in scripts/79 format.
Restartable at any point (finished pieces are skipped). Usage: python 79b_... [workers]."""
import importlib.util
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
PLAN_F, PART_F, ROWS_F = OUT / "modelb_split_plan.jsonl", OUT / "modelb_split_parts.jsonl", OUT / "modelb_draws.jsonl"
s = importlib.util.spec_from_file_location("m79", Path(__file__).with_name("79_p2v4_modelb_draws.py"))
m79 = importlib.util.module_from_spec(s); s.loader.exec_module(m79)
_W = {}


def _setup(name, K, draw):
    from shield.utils.config import Paths
    kr = _W.setdefault("kr", m79._load57())
    D, _ = kr.instance(name, Paths(False))
    n = len(D); L = np.random.default_rng(5000 + 100 * draw + n).uniform(100, 400, n)
    return kr, D, L, n


def plan_task(t):
    name, K, draw = t
    kr, D, L, n = _setup(name, K, draw)
    kr.U = m79.U0
    p0, plan = m79.pmin(D, L, K, m79.U0, kr, math.ceil(K * L.sum() / m79.U0))
    return {"topology": name, "K": K, "draw": draw, "n": n, "planned_p": p0, "C0": list(map(int, plan["C"])), "z0": plan["z"]}


def part_task(t):
    pl, key, ratio = t
    kr, D, L, n = _setup(pl["topology"], pl["K"], pl["draw"])
    U = m79.U0 * ratio; kr.U = U; K = pl["K"]
    util = m79.loads_util(D, L, np.array(pl["C0"]), K, U)
    if util <= 1 + 1e-9:
        p, z = pl["planned_p"], pl["z0"]
    else:
        p, best = m79.pmin(D, L, K, U, kr, math.ceil(K * L.sum() / U)); z = best["z"] if best else None
    kr.U = m79.U0
    return {"topology": pl["topology"], "K": K, "draw": pl["draw"], "key": key,
            "res": {"capacity_kpps": U, "planned_feasible": util <= 1 + 1e-9, "planned_max_util": util, "p_min": p, "z_at_p_min": z}}


def _read(f):
    return [json.loads(l) for l in open(f)] if f.exists() else []


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ratios = {f"r{r:.2f}": r for r in m79.GRID}
    if (OUT / "capacity_v4.json").exists():
        ratios.update(json.load(open(OUT / "capacity_v4.json"))["modelb_ratio"])
    done = {(r["topology"], r["K"], r["draw"]) for r in _read(ROWS_F)}
    todo = [(n, K, d) for n in m79.NETS for K in m79.KS for d in range(m79.DRAWS) if (n, K, d) not in done]
    print(len(todo), "rows missing:", todo, flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        plans = {(p["topology"], p["K"], p["draw"]): p for p in _read(PLAN_F)}
        need = [t for t in todo if t not in plans]
        with open(PLAN_F, "a") as f:
            for p in pool.imap_unordered(plan_task, need):
                f.write(json.dumps(p) + "\n"); f.flush(); plans[(p["topology"], p["K"], p["draw"])] = p
                print("plan", p["topology"], p["K"], p["draw"], "p0", p["planned_p"], flush=True)
        parts = {(x["topology"], x["K"], x["draw"], x["key"]): x for x in _read(PART_F)}
        tasks = [(plans[t], k, r) for t in todo for k, r in ratios.items() if (*t, k) not in parts]
        print(len(tasks), "scenario tasks", flush=True)
        with open(PART_F, "a") as f:
            for x in pool.imap_unordered(part_task, tasks):
                f.write(json.dumps(x) + "\n"); f.flush(); parts[(x["topology"], x["K"], x["draw"], x["key"])] = x
                print("part", x["topology"], x["K"], x["draw"], x["key"], "p_min", x["res"]["p_min"], flush=True)
    for t in todo:
        p = plans[t]
        row = {"topology": t[0], "K": t[1], "draw": t[2], "n": p["n"], "planned_p": p["planned_p"],
               "scenarios": {k: parts[(*t, k)]["res"] for k in ratios}}
        with open(ROWS_F, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["topology"], row["K"], row["draw"], row["planned_p"], {k: v["p_min"] for k, v in row["scenarios"].items()}, flush=True)


if __name__ == "__main__":
    main()
