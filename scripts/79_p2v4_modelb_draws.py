"""P2 v4, experiment X5: the published FFCCP (Killi & Rao) under escalation shift, over 20 random load draws per network.

v3 (scripts/58 --part shift) used one load draw per network (seed 1000 + n) and K = 2. Here, for AT&T, GEANT and SURFNET,
K in {2, 3}, and 20 draws of switch loads U(100, 400) kpps (seeds 5000 + 100 * draw + n):
  planned p0 = smallest p whose ILP optimum is feasible at the planned capacity U = 7,800 kpps (e0 = 10%),
  planned set C0 = that ILP optimum; for each scenario e: U(e) = 7,800 * mu_c(e) / mu_c(0.10) with the capacity ratio
  of the cascade that produced the scenario (outputs_v5/p2/capacity_v4.json, from X1); planned set feasible?, its
  maximum utilisation, and p_min(e) = smallest p with a feasible ILP optimum (exact, scipy HiGHS).
Output: outputs_v5/p2/modelb_draws.jsonl (one line per network, K, draw)."""

import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from shield.utils.config import Paths
from shield.utils.io import append_jsonl, read_jsonl

OUT = Path("D:/shield_run/outputs_v5/p2"); OUT.mkdir(parents=True, exist_ok=True)
NETS, KS, DRAWS, U0 = ["AttMpls", "Geant2012", "Surfnet"], (2, 3), 20, 7800.0
_W = {}


def _load57():
    s = importlib.util.spec_from_file_location("kr57", Path(__file__).with_name("57_killi_rao.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def loads_util(D, L, C, K, U):
    load = np.zeros(len(C))
    for i in range(len(D)):
        order = np.argsort(D[i, C])
        for l in range(K):
            load[order[l]] += L[i]
    return float(load.max() / U)


def pmin(D, L, K, U, kr, p_lo):
    for p in range(max(p_lo, K), len(D)):
        r = kr.milp_ffccp(D, L, p, K, time_limit=1800)
        if r is not None and r["z"] < 1e3 and r["C"] and loads_util(D, L, np.array(r["C"]), K, U) <= 1 + 1e-9:
            return p, r
    return None, None


def _task(t):
    name, K, draw, ratios = t
    kr = _W.setdefault("kr", _load57())
    D, _ = kr.instance(name, Paths(False))
    n = len(D); L = np.random.default_rng(5000 + 100 * draw + n).uniform(100, 400, n)
    kr.U = U0                                       # milp_ffccp and ffccp_z read the module-level capacity U
    p0, plan = pmin(D, L, K, U0, kr, math.ceil(K * L.sum() / U0))   # every load counts K times: p U >= K sum L
    row = {"topology": name, "K": K, "draw": draw, "n": n, "planned_p": p0, "scenarios": {}}
    C0 = np.array(plan["C"])
    for key, ratio in ratios.items():
        U = U0 * ratio; kr.U = U
        util = loads_util(D, L, C0, K, U)
        p, best = (p0, plan) if util <= 1 + 1e-9 else pmin(D, L, K, U, kr, math.ceil(K * L.sum() / U))
        row["scenarios"][key] = {"capacity_kpps": U, "planned_feasible": util <= 1 + 1e-9, "planned_max_util": util,
                                 "p_min": p, "z_at_p_min": best["z"] if best else None}
    kr.U = U0
    return row


GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]           # capacity ratio U(e) / U(e0)


def main():
    """Capacity ratios: a fixed grid (p_min is monotone in the ratio, so any scenario maps onto it), plus the measured
    scenario ratios of X1 when outputs_v5/p2/capacity_v4.json exists."""
    ratios = {f"r{r:.2f}": r for r in GRID}
    if (OUT / "capacity_v4.json").exists():
        ratios.update(json.load(open(OUT / "capacity_v4.json"))["modelb_ratio"])
    out_f = OUT / "modelb_draws.jsonl"
    done = {(r["topology"], r["K"], r["draw"]) for r in read_jsonl(out_f)} if out_f.exists() else set()
    tasks = [(n, K, d, ratios) for n in NETS for K in KS for d in range(DRAWS) if (n, K, d) not in done]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.imap_unordered(_task, tasks):
            append_jsonl(row, out_f)
            print(row["topology"], row["K"], row["draw"], row["planned_p"],
                  {k: v["p_min"] for k, v in row["scenarios"].items()}, flush=True)


if __name__ == "__main__":
    main()
