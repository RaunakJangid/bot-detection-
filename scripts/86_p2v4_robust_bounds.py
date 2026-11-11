"""P2 v4, Option A follow-up: valid bounds on the number of controllers for every plan of scripts/84.

Several ILPs of scripts/84 stopped at the 30 min time limit; when no feasible solution was found at p, the loop moved
to p + 1, so the reported p is an upper bound, not a proven optimum (seen as chance-constrained plans needing more
controllers than the robust plan, which is impossible). This script reports, per network and plan:
  LB  capacity bound: the smallest p for which at least ceil((1 - beta) S) planning scenarios satisfy
      p x U_s >= K x sum_i L_s,i (every controller set with fewer controllers violates more scenarios);
  UB  the smallest p of any placement found by any plan (ILP solutions of scripts/84 or the search below) that meets
      this plan's in-sample constraint (at most floor(beta S) violated planning scenarios);
  search  simulated annealing over controller sets with swap moves for each p from LB upward until the constraint is
      met: objective = violated scenarios beyond the allowance (x 1e3) + worst-case latency z; 20 restarts x 20,000
      evaluations; every scenario evaluated exactly (closest assignment fixes the loads).
Exact when LB = UB, or when the ILP of scripts/84 proved optimality (status 0) at p and infeasibility below.
The held-out evaluation is repeated for the placement at UB. Output: outputs_v5/p2/robust_bounds.json."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
U0, K = 7800.0, 2


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


m84 = _load("84_p2v4_robust_placement.py")
kr = _load("57_killi_rao.py")
rng = np.random.default_rng(86)


def assign(D, C):
    """n x p 0/1 matrix: switch i references its K closest controllers in C."""
    C = np.asarray(C); A = np.zeros((len(D), len(C)))
    o = np.argsort(D[:, C], axis=1)[:, :K]
    for l in range(K):
        A[np.arange(len(D)), o[:, l]] = 1
    return A


def violated(D, C, Ls, Us):
    """number of (load draw, capacity) pairs with an overloaded controller."""
    mx = (np.stack(Ls) @ assign(D, C)).max(1)                    # max controller load per draw
    return int((mx[:, None] > np.asarray(Us)[None, :] + 1e-9).sum())


def zval(D, C):
    return float(np.sort(D[:, np.asarray(C)], axis=1)[:, K - 1].max())


def sa(D, p, Ls, Us, allow, restarts=20, evals=20000):
    n = len(D); best = None
    for _ in range(restarts):
        C = list(rng.choice(n, p, replace=False)); cur = 1e3 * max(0, violated(D, C, Ls, Us) - allow) + zval(D, C)
        T = 0.05 * cur + 1e-6
        for it in range(evals // restarts):
            j = rng.integers(p); v = int(rng.integers(n))
            if v in C:
                continue
            C2 = C.copy(); C2[j] = v
            f = 1e3 * max(0, violated(D, C2, Ls, Us) - allow) + zval(D, C2)
            if f < cur or rng.random() < math.exp((cur - f) / T):
                C, cur = C2, f
            T *= 0.9995
            if best is None or cur < best[0]:
                best = (cur, sorted(C))
    return best


def main():
    R = json.load(open(OUT / "robust_placement.json"))
    nocap, cap = R["scenarios"]["nocap"], R["scenarios"]["cap"]
    Unc, Ucap = [U0 * r for _, _, r in nocap], [U0 * r for _, _, r in cap]
    out = {}
    for nm, res in R["nets"].items():
        D, _ = kr.instance(nm, Paths(False)); n = len(D)
        Lp, Lt = m84.loads(n, range(10)), m84.loads(n, range(10, 20))
        spec = {"nominal": ([Lp[0]], [U0], 0.0), "robust": (Lp, Unc, 0.0), "cc05": (Lp, Unc, 0.05),
                "cc10": (Lp, Unc, 0.10), "cap_cc05": (Lp, Ucap, 0.05)}
        cands = [np.asarray(r["C"]) for r in res.values() if r and r.get("C")]
        out[nm] = {}
        for dn, (Ls, Us, beta) in spec.items():
            S = len(Ls) * len(Us); allow = math.floor(beta * S + 1e-9)
            need = S - allow
            tot = sorted(K * L.sum() / U for L in Ls for U in Us)              # per-scenario minimum p (real)
            lb = max(K, math.ceil(tot[need - 1] - 1e-9))
            ub, ubC, src = None, None, None
            for C in cands:
                if violated(D, C, Ls, Us) <= allow and (ub is None or len(C) < ub or (len(C) == ub and zval(D, C) < zval(D, ubC))):
                    ub, ubC, src = len(C), C, "ilp"
            for p in range(lb, (ub or n) + 1):
                if ub is not None and p >= ub:
                    break
                b = sa(D, p, Ls, Us, allow)
                if b and b[0] < 1e3:
                    ub, ubC, src = p, np.asarray(b[1]), "search"; break
            ilp = res.get(dn) or {}
            exact = (ub == lb) or (ilp.get("status") == 0 and ilp.get("p") == ub)
            r = {"lb": lb, "ub": ub, "exact": bool(exact), "ub_source": src, "ilp_p": ilp.get("p"), "ilp_status": ilp.get("status"),
                 "z_at_ub": zval(D, ubC) if ubC is not None else None, "C_at_ub": None if ubC is None else ubC.tolist(),
                 "in_sample_violations": None if ubC is None else violated(D, ubC, Ls, Us), "allowed": allow, "S": S}
            if ubC is not None:
                r["heldout_nocap"] = violated(D, ubC, Lt, Unc) / (len(Lt) * len(Unc))
                r["heldout_cap"] = violated(D, ubC, Lt, Ucap) / (len(Lt) * len(Ucap))
                r["heldout_nominal"] = violated(D, ubC, Lt, [U0]) / len(Lt)
            out[nm][dn] = r
            print(nm, dn, {k: v for k, v in r.items() if k != "C_at_ub"}, flush=True)
    json.dump(out, open(OUT / "robust_bounds.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
