"""P2 v4, experiment X10: separate detection servers (the decoupled design of Sapkota et al.) against detection on the
controllers (this paper), in machines and latency. CPU only.

Same machine type in both designs (m cores). Co-located: a controller gives a share eta = 0.5 of its cores to the
cascade, capacity U(e) = 7,800 x mu_c(e) / mu_c(0.10) kpps; counts from scripts/87 (robust_final.json: robust over
load x escalation scenarios, without and with the online cap).
Separate: controllers do control only; their count is set to K = 2 (the least the failure-foresight model allows,
which favours the baseline). Detection runs on a pool of n servers with all cores for the cascade, capacity 2 U(e)
each, perfectly load-balanced (favours the baseline), plus one spare server (N + 1, the analogue of the K = 2
reference reserve): n = ceil(max_s sum_i L_s,i / (2 U(e_s))) + 1 over the planning scenarios (load draws 0 to 9 x
escalation set), checked on the held-out draws 10 to 109. Detection sites: p-center ILP (minimise the largest
switch-to-server distance z_det, scipy milp). Every new flow travels switch -> detection server -> controller, so
the added worst-case delay is max_i [d(i, det(i)) + d(det(i), c(i)) - d(i, c(i))], c(i) = closest of the K = 2
controllers (controllers at the p-center sites for p = 2).
Output: outputs_v5/p2/separate_server.json."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
U0, K = 7800.0, 2


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


kr = _load("57_killi_rao.py"); m84 = _load("84_p2v4_robust_placement.py")


def pcenter(D, p):
    """exact p-center: sites y_j, assignment x_ij, minimise z >= sum_j D_ij x_ij."""
    n = len(D); nv = n + n * n + 1; iz = nv - 1
    rows, cols, vals, lo, hi = [], [], [], [], []; r = 0
    def add(cs, vs, a, b):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs); lo.append(a); hi.append(b); r += 1
    add(list(range(n)), [1] * n, p, p)
    for i in range(n):
        add([n + i * n + j for j in range(n)], [1] * n, 1, 1)
        add([n + i * n + j for j in range(n)] + [iz], list(D[i]) + [-1], -np.inf, 0)
        for j in range(n):
            add([n + i * n + j, j], [1, -1], -np.inf, 0)
    c = np.zeros(nv); c[iz] = 1; integ = np.ones(nv); integ[iz] = 0; ub = np.ones(nv); ub[iz] = np.inf
    res = milp(c, constraints=LinearConstraint(coo_matrix((vals, (rows, cols)), shape=(r, nv)), lo, hi),
               integrality=integ, bounds=Bounds(np.zeros(nv), ub), options={"time_limit": 600})
    return np.flatnonzero(res.x[:n] > 0.5), float(res.fun)


def main():
    R = json.load(open(OUT / "robust_placement.json")); F = json.load(open(OUT / "robust_final.json"))
    sc = {"nocap": [r for _, _, r in R["scenarios"]["nocap"]], "cap": [r for _, _, r in R["scenarios"]["cap"]]}
    out = {}
    for nm in ("AttMpls", "Geant2012", "Surfnet"):
        D, _ = kr.instance(nm, Paths(False)); n = len(D)
        Lp, Lt = m84.loads(n, range(10)), m84.loads(n, range(10, 110))
        ctrl, z_ctrl = pcenter(D, K)
        out[nm] = {"n": n, "controllers_separate": K, "z_ctrl_pcenter": z_ctrl}
        for reg, rs in sc.items():
            need = max(L.sum() / (2 * U0 * r) for L in Lp for r in rs)
            nd = math.ceil(need - 1e-9) + 1
            held = float(np.mean([L.sum() / (2 * U0 * r) > nd - 1 + 1e-9 for L in Lt for r in rs]))
            sites, zdet = pcenter(D, nd)
            ci = ctrl[np.argmin(D[:, ctrl], axis=1)]; di = sites[np.argmin(D[:, sites], axis=1)]
            extra = float(max(D[i, di[i]] + D[di[i], ci[i]] - D[i, ci[i]] for i in range(n)))
            co = {"nocap": F[nm]["robust"], "cap": F[nm]["cap_robust"]}[reg]
            out[nm][reg] = {"detection_servers": nd, "machines_separate": K + nd, "heldout_overload_with_spare": held,
                            "z_det": zdet, "added_delay_worst": extra,
                            "machines_colocated": [co["lb"], co["ub"]], "z_colocated": co["z"]}
            print(nm, reg, out[nm][reg], flush=True)
    json.dump(out, open(OUT / "separate_server.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
