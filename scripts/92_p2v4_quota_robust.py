"""P2 v4, Option A with the admission quota: robust placement when escalation is bounded per window.

X12 (scripts/88) shows that the online cap plus the admission quota keeps every 1,000-flow block at or below the
planned 10% share (worst-block capacity ratio 1.02 to 1.09 in every flooding run and without attack). The capacity
of Step 1 then never falls below U0 = 7,800 kpps, and the robust plan only has to cover the load: scenarios = 10
planning load draws x {U0}, beta = 0. Bounds and closing as scripts/86 and 87 (capacity LB; UB from every placement
of robust_final.json that fits, or the search; feasibility ILP at UB - 1, 2 h). Held-out: draws 10 to 109 at U0.
Output: outputs_v5/p2/quota_robust.json."""
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


m87 = _load("87_p2v4_robust_close.py"); m86, m84, kr = m87.m86, m87.m84, m87.kr


def main():
    F = json.load(open(OUT / "robust_final.json")); out = {}
    for nm in ("AttMpls", "Geant2012", "Surfnet"):
        D, _ = kr.instance(nm, Paths(False)); n = len(D)
        Ls, Us = m84.loads(n, range(10)), [m86.U0]; Lt = m84.loads(n, range(10, 110))
        lb = max(m86.K, max(math.ceil(m86.K * L.sum() / m86.U0 - 1e-9) for L in Ls))
        ub, C = None, None
        for r in F[nm].values():
            c = np.asarray(r["C"])
            if m86.violated(D, c, Ls, Us) == 0 and (ub is None or len(c) < ub):
                ub, C = len(c), c.tolist()
        for p in range(lb, ub or n):
            b = m86.sa(D, p, Ls, Us, 0)
            if b and b[0] < 1e3:
                ub, C = p, b[1]; break
        steps, proven, p = [], ub == lb, ub - 1
        while p >= lb and not proven:
            t = time.time(); m87.LAST.clear()
            res = m84.milp_cc(D, [L for L in Ls for _ in Us], [U for _ in Ls for U in Us], p, 0.0)
            st = m87.LAST.get("status"); steps.append({"p": p, "status": st, "seconds": round(time.time() - t, 1)})
            if res is not None and st == 0:
                ub, C = p, res["C"]; p -= 1; proven = ub == lb
            elif st == 2:
                proven = True
            else:
                break
        Cn = np.asarray(C)
        out[nm] = {"lb": lb, "ub": ub, "exact": bool(proven), "C": C, "steps": steps, "z": m86.zval(D, Cn),
                   "heldout_overload_U0": m86.violated(D, Cn, Lt, Us) / len(Lt),
                   "nominal_p": F[nm]["nominal"]["ub"], "cap_robust_p": F[nm]["cap_robust"]["ub"]}
        print(nm, {k: v for k, v in out[nm].items() if k != "C"}, flush=True)
    json.dump(out, open(OUT / "quota_robust.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
