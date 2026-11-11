"""P2 v4, Option A closing step: proves or tightens the controller counts left open by scripts/86 and re-tests every
final placement on 100 held-out load draws.

1. Open cases (LB < UB in robust_bounds.json): feasibility version of the ILP of scripts/84 (same constraints, zero
   objective, so the solver stops at the first feasible point) at p = UB - 1, then UB - 2, ... down to LB, 2 h limit each.
   Infeasible (status 2) proves UB optimal; feasible lowers UB; time limit leaves [LB, UB].
2. New plan cap_robust: every (load draw, capacity under the online cap) pair must hold (beta = 0); LB, UB by the
   capacity bound and the search of scripts/86, then step 1.
3. Held-out: load draws 10 to 109 (100 draws) x every capacity scenario, closest assignment, for the placement at UB.
Each (network, plan) runs as its own process (--one). Output: outputs_v5/p2/robust_final.json."""
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp as _milp

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
TL = 7200


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


m84 = _load("84_p2v4_robust_placement.py")
m86 = _load("86_p2v4_robust_bounds.py")
kr = _load("57_killi_rao.py")
LAST = {}


def _feas(c, **kw):                                   # zero objective: feasibility only
    kw["options"] = {**kw.get("options", {}), "time_limit": TL}
    res = _milp(np.zeros_like(c), **kw); LAST["status"] = int(res.status)
    return res


m84.milp = _feas


def spec(nm):
    R = json.load(open(OUT / "robust_placement.json"))
    U0 = m86.U0
    Unc = [U0 * r for _, _, r in R["scenarios"]["nocap"]]; Ucap = [U0 * r for _, _, r in R["scenarios"]["cap"]]
    D, _ = kr.instance(nm, Paths(False)); n = len(D)
    Lp = m84.loads(n, range(10))
    return D, Lp, Unc, Ucap, {"nominal": ([Lp[0]], [U0], 0.0), "robust": (Lp, Unc, 0.0), "cc05": (Lp, Unc, 0.05),
                              "cc10": (Lp, Unc, 0.10), "cap_cc05": (Lp, Ucap, 0.05), "cap_robust": (Lp, Ucap, 0.0)}


def one(nm, dn):
    D, Lp, Unc, Ucap, S = spec(nm); Ls, Us, beta = S[dn]
    B = json.load(open(OUT / "robust_bounds.json"))[nm]
    nS = len(Ls) * len(Us); allow = math.floor(beta * nS + 1e-9)
    if dn in B:
        lb, ub, C = B[dn]["lb"], B[dn]["ub"], B[dn]["C_at_ub"]
    else:                                              # new plan: bounds as scripts/86
        tot = sorted(m86.K * L.sum() / U for L in Ls for U in Us)
        lb = max(m86.K, math.ceil(tot[nS - allow - 1] - 1e-9)); ub = C = None
        cands = [np.asarray(v["C_at_ub"]) for v in B.values() if v.get("C_at_ub")]
        for c in cands:
            if m86.violated(D, c, Ls, Us) <= allow and (ub is None or len(c) < ub):
                ub, C = len(c), c.tolist()
        for p in range(lb, (ub or len(D))):
            b = m86.sa(D, p, Ls, Us, allow)
            if b and b[0] < 1e3:
                ub, C = p, b[1]; break
    steps = []
    Lpairs = [L for L in Ls for _ in Us]; Upairs = [U for _ in Ls for U in Us]
    p = ub - 1; proven = ub == lb
    while p >= lb and not proven:
        t = time.time(); LAST.clear(); res = m84.milp_cc(D, Lpairs, Upairs, p, beta)
        st = LAST.get("status"); steps.append({"p": p, "status": st, "seconds": round(time.time() - t, 1)})
        print(nm, dn, steps[-1], flush=True)
        if res is not None and st == 0:
            ub, C = p, res["C"]; p -= 1; proven = ub == lb
        elif st == 2:
            proven = True
        else:
            break
    return {"lb": lb, "ub": ub, "exact": bool(proven), "C": C, "steps": steps, "allowed": allow, "S": nS,
            "in_sample_violations": m86.violated(D, np.asarray(C), Ls, Us)}


def main():
    nets = ["AttMpls", "Geant2012", "Surfnet"]
    B = json.load(open(OUT / "robust_bounds.json"))
    jobs = [(nm, dn) for nm in nets for dn in list(B[nm]) + ["cap_robust"]]
    parts = OUT / "robust_final_parts"; parts.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pending, running = [j for j in jobs if not (parts / f"{j[0]}_{j[1]}.json").exists()], []
    while pending or running:
        while pending and len(running) < workers:
            nm, dn = pending.pop(0)
            running.append((nm, dn, subprocess.Popen([sys.executable, __file__, "--one", nm, dn], stdin=subprocess.DEVNULL,
                                                     stdout=open(parts / f"{nm}_{dn}.log", "w"), stderr=subprocess.STDOUT)))
        time.sleep(15)
        running = [r for r in running if r[2].poll() is None]
    out = {}
    for nm in nets:
        D, Lp, Unc, Ucap, S = spec(nm); n = len(D); Lt = m84.loads(n, range(10, 110)); out[nm] = {}
        for dn in list(B[nm]) + ["cap_robust"]:
            r = json.load(open(parts / f"{nm}_{dn}.json"))
            C = np.asarray(r["C"])
            r.update(z=m86.zval(D, C), heldout_draws=len(Lt),
                     heldout_nocap=m86.violated(D, C, Lt, Unc) / (len(Lt) * len(Unc)),
                     heldout_cap=m86.violated(D, C, Lt, Ucap) / (len(Lt) * len(Ucap)),
                     heldout_nominal=m86.violated(D, C, Lt, [m86.U0]) / len(Lt))
            out[nm][dn] = r
            print(nm, dn, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("C", "steps")}, flush=True)
    json.dump(out, open(OUT / "robust_final.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--one":
        json.dump(one(sys.argv[2], sys.argv[3]), open(OUT / "robust_final_parts" / f"{sys.argv[2]}_{sys.argv[3]}.json", "w"), indent=1)
    else:
        main()
