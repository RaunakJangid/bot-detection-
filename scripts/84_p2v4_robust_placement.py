"""P2 v4, Option A: chance-constrained detection-aware controller placement on the published FFCCP model. CPU only.

Uncertainty: (i) switch loads, 20 draws U(100, 400) kpps per network (seeds of scripts/79), split 10 for planning and 10
held out; (ii) the escalation share of the controllers' cascade, from the measured scenarios of scripts/85
(outputs_v5/p2/escalation_scenarios.json): without the cap, the per-seed share e of the three shifted settings
(CICIoMT2024 6-class and 19-class, IoT-23 1-1); with the online cap, the shares of 1,000-flow blocks under the running
quantile (same settings). Each e becomes a capacity U(e) = 7,800 x mu_c(e) / mu_c(0.10) kpps with the measured cost of
the cascade that produced it (outputs_v5/p2/throughput_v4.json, batch 256). A scenario s = (load draw, e).

Plans (K = 2 references per switch, exact ILP of Killi & Rao extended with one capacity row per scenario; closest
assignment is fixed by the controller set, so all scenarios share one assignment):
  nominal      U = 7,800 (e0 = 10%), the first planning draw only: what an operator plans today;
  robust       capacity feasible in every planning scenario (no cap);
  cc05, cc10   chance-constrained: feasible in at least 95% / 90% of planning scenarios (binary w_s with big-M);
  cap_cc05     scenarios under the online cap, at least 95% feasible.
For each plan: the smallest p with a feasible ILP (then minimum worst-case latency z), and the held-out evaluation over
the 10 test draws x the scenario set of its regime (and of the other regime): share of scenarios with an overloaded
controller, maximum utilisation.
Output: outputs_v5/p2/robust_placement.json."""

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
U0, K, DRAWS, NETS = 7800.0, 2, 20, ["AttMpls", "Geant2012", "Surfnet"]
TL = 1800


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def scenarios():
    T = json.load(open(OUT / "throughput_v4.json")); E = json.load(open(OUT / "escalation_scenarios.json"))
    casc = {"ciciomt_category": ("ciciomt_category", "teacher"), "ciciomt_attack": ("ciciomt_attack", "teacher"),
            "iot23_capture_shift_CTU-IoT-Malware-Capture-1-1": ("iot23_capture_shift", "ensemble")}
    nocap, cap = [], []
    for st, (ck, ex) in casc.items():
        ms, mt = T[ck]["256"]["student_fp32"]["throughput"], T[ck]["256"][ex]["throughput"]
        ratio = lambda e: (1 / ms + 0.10 / mt) / (1 / ms + e / mt)
        nocap += [(st, float(e), ratio(e)) for e in E[st]["e_nocap"]]
        cap += [(st, float(b), ratio(b)) for bl in E[st]["blocks_online"] for b in bl]
    # thin the cap blocks to at most 60 evenly spaced quantiles (keeps the ILP size bounded, preserves the tail)
    if len(cap) > 30:      # thin to 30 evenly spaced quantiles of the block shares (bounded ILP size, tail kept)
        cs = sorted(cap, key=lambda x: x[2]); idx = np.unique(np.round(np.linspace(0, len(cs) - 1, 30)).astype(int))
        cap = [cs[i] for i in idx]
    return nocap, cap


def loads(n, draws):
    return [np.random.default_rng(5000 + 100 * d + n).uniform(100, 400, n) for d in draws]


def milp_cc(D, Ls, Us, p, beta):
    """Killi & Rao ILP with one capacity row per scenario (L_s, U_s); at least ceil((1 - beta) S) rows must hold."""
    n, S = len(D), len(Ls)
    iy = lambda j: j
    ir = lambda l, i, j: n + (l * n + i) * n + j
    iz = n + K * n * n
    iw = lambda s: iz + 1 + s
    nv = iz + 1 + S
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(cs, vs, a, b):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs); lo.append(a); hi.append(b); r += 1

    ub = np.ones(nv); ub[iz] = np.inf
    for i in range(n):
        for l in range(K):
            add([ir(l, i, j) for j in range(n)], [1] * n, 1, 1)
            if l > 0:
                ub[ir(l, i, i)] = 0
    add([iy(j) for j in range(n)], [1] * n, p, p)
    for i in range(n):
        for j in range(n):
            add([ir(l, i, j) for l in range(K)] + [iy(j)], [1] * K + [-1], -np.inf, 0)
            closer = np.flatnonzero(D[i] <= D[i, j])
            for l in range(K):
                cs = [ir(l, i, a) for a in closer] + [ir(m, i, j) for m in range(l)] + [iy(j)]
                add(cs, [1] * (len(closer) + l) + [-1], 0, np.inf)
        add([ir(K - 1, i, j) for j in range(n)] + [iz], list(D[i]) + [-1], -np.inf, 0)
    for s, (L, U) in enumerate(zip(Ls, Us)):
        M = K * float(L.sum())
        for j in range(n):    # sum_i sum_l L_i r_lij - U y_j <= M (1 - w_s)
            add([ir(l, i, j) for l in range(K) for i in range(n)] + [iy(j), iw(s)],
                [L[i] for l in range(K) for i in range(n)] + [-U, M], -np.inf, M)
    need = math.ceil((1 - beta) * S - 1e-9)
    add([iw(s) for s in range(S)], [1] * S, need, S)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv))
    c = np.zeros(nv); c[iz] = 1
    integ = np.ones(nv); integ[iz] = 0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": TL, "disp": False})
    if res.x is None:
        return None
    return {"z": float(res.fun), "status": int(res.status), "C": np.flatnonzero(res.x[:n] > 0.5).tolist(),
            "gap": float(getattr(res, "mip_gap", float("nan")) or 0.0)}


def util(D, L, C, U):
    C = np.asarray(C); load = np.zeros(len(C))
    for i in range(len(D)):
        o = np.argsort(D[i, C])
        for l in range(K):
            load[o[l]] += L[i]
    return float(load.max() / U)


def plan(D, Ls, Us, beta, p_lo):
    for p in range(max(K, p_lo), len(D)):
        t = time.time(); res = milp_cc(D, Ls, Us, p, beta)
        if res is not None and res["status"] in (0, 1) and res["z"] < 1e3:
            res.update(p=p, seconds=time.time() - t); return res
    return None


def evaluate(D, C, Ls, Us):
    us = [util(D, L, C, U) for L in Ls for U in Us]
    return {"violation_share": float(np.mean(np.array(us) > 1 + 1e-9)), "max_util": float(max(us)),
            "median_util": float(np.median(us)), "n": len(us)}


DESIGNS = ("nominal", "robust", "cc05", "cc10", "cap_cc05")


def _design(name, dn, nocap, cap):
    kr = _load("57_killi_rao.py"); D, _ = kr.instance(name, Paths(False)); n = len(D)
    Lp = loads(n, range(10))
    Unc, Ucap = [U0 * r for _, _, r in nocap], [U0 * r for _, _, r in cap]
    Ls, Us, beta = {"nominal": ([Lp[0]], [U0], 0.0), "robust": (Lp, Unc, 0.0), "cc05": (Lp, Unc, 0.05),
                    "cc10": (Lp, Unc, 0.10), "cap_cc05": (Lp, Ucap, 0.05)}[dn]
    Lpairs = [L for L in Ls for _ in Us]; Upairs = [U for _ in Ls for U in Us]
    p_lo = math.ceil(K * min(L.sum() for L in Ls) / max(Us))
    t = time.time(); pl = plan(D, Lpairs, Upairs, beta, p_lo)
    return name, dn, (None if pl is None else {**pl, "seconds": time.time() - t, "n_scen_plan": len(Lpairs), "beta": beta})


def _task(args):
    return _design(*args)


def in_sample_ok(D, C, Lpairs, Upairs, beta):
    v = [util(D, L, C, U) > 1 + 1e-9 for L, U in zip(Lpairs, Upairs)]
    return sum(v) <= math.floor(beta * len(v) + 1e-9)


def main():
    import multiprocessing as mp
    nets = [a for a in sys.argv[1:] if not a.isdigit()] or NETS
    workers = int(next((a for a in sys.argv[1:] if a.isdigit()), 10))
    nocap, cap = scenarios()
    out_f = OUT / "robust_placement.json"
    out = json.load(open(out_f)) if out_f.exists() else {}
    out.setdefault("scenarios", {"nocap": nocap, "cap": cap}); out.setdefault("nets", {})
    tasks = [(nm, dn) for nm in nets for dn in DESIGNS if dn not in out["nets"].get(nm, {})]
    print(f"{len(tasks)} (network, plan) solves", flush=True)
    # one independent process per solve (a multiprocessing pool failed to start workers in a background job on
    # Windows: WinError 6 while importing asyncio); each child writes outputs_v5/p2/robust_parts/<net>_<plan>.json
    import subprocess
    parts = OUT / "robust_parts"; parts.mkdir(exist_ok=True)
    pending, running = list(tasks), []
    while pending or running:
        while pending and len(running) < workers:
            nm, dn = pending.pop(0)
            log = open(parts / f"{nm}_{dn}.log", "w")
            running.append((nm, dn, subprocess.Popen([sys.executable, __file__, "--one", nm, dn], stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(20)
        for item in list(running):
            nm, dn, pr = item
            if pr.poll() is not None:
                running.remove(item)
                f = parts / f"{nm}_{dn}.json"
                pl = json.load(open(f)) if f.exists() else None
                out["nets"].setdefault(nm, {})[dn] = pl if pl else {"p": None, "error": f"exit {pr.returncode}"}
                json.dump(out, open(out_f, "w"), indent=1)
                print(nm, dn, None if not pl else {k: v for k, v in pl.items() if k != "C"}, flush=True)
    # incumbent check and held-out evaluation, per network (placements of other plans with the same p, if they meet
    # this plan's in-sample constraint, can replace a time-limited solution with a lower worst-case latency z)
    kr = _load("57_killi_rao.py")
    for nm, res in out["nets"].items():
        D, _ = kr.instance(nm, Paths(False)); n = len(D)
        Lp, Lt = loads(n, range(10)), loads(n, range(10, 20))
        Unc, Ucap = [U0 * r for _, _, r in nocap], [U0 * r for _, _, r in cap]
        spec = {"nominal": ([Lp[0]], [U0], 0.0), "robust": (Lp, Unc, 0.0), "cc05": (Lp, Unc, 0.05),
                "cc10": (Lp, Unc, 0.10), "cap_cc05": (Lp, Ucap, 0.05)}
        for dn, r in res.items():
            if not r or r.get("p") is None:
                continue
            Ls, Us, beta = spec[dn]; Lpairs = [L for L in Ls for _ in Us]; Upairs = [U for _ in Ls for U in Us]
            zf = lambda C: float(max(np.sort(D[i, C])[K - 1] for i in range(n)))
            best_C, best_z, src = r["C"], zf(np.asarray(r["C"])), dn
            for on, o in res.items():
                if o and o.get("p") == r["p"] and on != dn:
                    C = np.asarray(o["C"])
                    if in_sample_ok(D, C, Lpairs, Upairs, beta) and zf(C) < best_z - 1e-9:
                        best_C, best_z, src = o["C"], zf(C), on
            r.update(C_final=best_C, z_final=best_z, z_source=src,
                     heldout_nocap=evaluate(D, best_C, Lt, Unc), heldout_cap=evaluate(D, best_C, Lt, Ucap),
                     heldout_nominal=evaluate(D, best_C, Lt, [U0]))
            print(nm, dn, "p", r["p"], "z", round(best_z, 3), "from", src, "| held-out viol nocap",
                  round(r["heldout_nocap"]["violation_share"], 3), "cap", round(r["heldout_cap"]["violation_share"], 3), flush=True)
    json.dump(out, open(out_f, "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--one":
        nc, cp = scenarios()
        nm, dn, pl = _design(sys.argv[2], sys.argv[3], nc, cp)
        (OUT / "robust_parts").mkdir(exist_ok=True)
        json.dump(pl, open(OUT / "robust_parts" / f"{nm}_{dn}.json", "w"), indent=1)
    else:
        main()
