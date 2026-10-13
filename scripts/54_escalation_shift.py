"""P2: controller placement when the detector's escalation load shifts with the traffic.

Each controller runs the edge cascade (tiny student; uncertain flows escalated to the expert). A flow costs
1/mu_student + e/mu_expert, so the controller service rate depends on the escalation share e. e is set on
validation (an entropy threshold that escalates b of validation flows) but on shifted traffic it changes.

Part A (measured escalation, no training): for every shift setting with saved model outputs, the student's entropy
threshold is fixed on validation for budgets b in {5, 10, 20}%, then on test we record
  e_test      the share actually escalated (label-free observable)
  f1_planned  cascade macro-F1 with the validation threshold (escalation e_test)
  f1_capped   cascade macro-F1 when escalation is capped at b on test (threshold = test-entropy quantile; label-free)
  f1_student / f1_expert  for reference.
  Settings: CICIoT 34-class (no shift, control), CICIoMT binary/category/attack (official capture split), IoT-23
  per held-out capture (same-family and capture-shift designs, binary; student small_mlp_kd, expert ensemble).
  Students: shield_s0-4 (CIC) / small_mlp_kd_s0-4 (IoT-23); expert: teacher_s0 / ensemble_s0.
Part B (placement): topologies and SLA rule of configs/coupled.yaml (Geant2012, Cogentco, syn200; rho 0.5/0.7/0.9;
  demand = rho * k_ref * mu_c(teacher); 95% of switches within SLA, also after the worst controller failure).
  Placement search: gwo_sa (best published method in our benchmark), tuned settings, 3 runs x 10,000 evaluations.
  For every e in the measured scenario set (median e_test per setting at the 10% budget) plus the planned e = 0.10:
    k_min(e)  fewest controllers meeting the SLA;
    the PLANNED placement (k_min at e = 0.10) re-evaluated at e: feasible?, % switches within SLA, overload.
  Mitigations compared per scenario: (i) add controllers (k_min(e) - k_min(0.10)); (ii) cap escalation at 10%
  (no extra controllers; macro-F1 loss = f1_planned - f1_capped from Part A)."""

import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.placement.coupling import feasible, topology_sla
from shield.placement.objective import make_demand
from shield.placement.runner import (DEMAND_SEED, build_problem, controller_mu, run_algorithm, topology_and_latency,
                                     tuned_cfg, with_params)
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, load_json, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("escshift")
BUDGETS = (0.05, 0.10, 0.20)
ALG = "gwo_sa"
_W: dict = {}


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def cascade_stats(Ps_v, Ps_t, pe_t, y_t, ps_t, binary=False):
    Hv, Ht = entropy(Ps_v), entropy(Ps_t)
    avg = "binary" if binary else "macro"
    out = {"f1_student": float(100 * f1_score(y_t, ps_t, average=avg)), "f1_expert": float(100 * f1_score(y_t, pe_t, average=avg))}
    for b in BUDGETS:
        t = np.quantile(Hv, 1 - b)
        esc = Ht > t
        pred = np.where(esc, pe_t, ps_t)
        tc = np.quantile(Ht, 1 - b)
        pred_c = np.where(Ht > tc, pe_t, ps_t)
        out[f"b{int(100 * b)}"] = {"e_test": float(esc.mean()), "f1_planned": float(100 * f1_score(y_t, pred, average=avg)),
                                    "f1_capped": float(100 * f1_score(y_t, pred_c, average=avg))}
    return out


def part_a(P: Paths) -> list:
    rows = []
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        yt = d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"
        Pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)); pe = Pe.argmax(1)
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))
            r = cascade_stats(Pv, Pt, pe, yt, Pt.argmax(1))
            rows.append({"setting": f"{ds}_{task}", "shift": ds != "ciciot", "seed": s, **r})
        log.info("%s_%s: e_test at 10%% budget %s", ds, task, [round(x["b10"]["e_test"], 3) for x in rows[-5:]])
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz")
        models = list(Z["models"]); yv, yt, cap = Z["yv"], Z["yt"], Z["capture"]
        pe_all = (Z["st"][models.index("ensemble_s0")] > 0.5).astype(int)
        two = lambda p: np.c_[1 - p, p]
        for c in np.unique(cap):
            m = cap == c
            if (yt[m] == 0).sum() < 100 or (yt[m] == 1).sum() < 100:
                continue
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                sv, st = Z["sv"][i], Z["st"][i][m]
                r = cascade_stats(two(sv), two(st), pe_all[m], yt[m], (st > 0.5).astype(int), binary=True)
                rows.append({"setting": f"iot23_{design}_{c}", "shift": True, "seed": s, **r})
            log.info("iot23 %s %s: e_test at 10%% budget %s", design, c, [round(x["b10"]["e_test"], 3) for x in rows[-5:]])
    return rows


def _init(tuned):
    _W["pcfg"] = load_config("placement", False)
    _W["ccfg"] = load_config("coupled", False)
    _W["paths"] = Paths(False)
    _W["tuned"] = tuned


def _best(topo, D, k, total, mu, pcfg, ccfg):
    problem = build_problem(topo, D, k, total, mu, pcfg)
    best = None
    for seed in range(ccfg["runs"]):
        res = run_algorithm(ALG, problem, topo, pcfg, seed, budget=ccfg["budget"])
        if best is None or res.best_F < best.best_F:
            best = res
    return problem, best


def _task(task):
    """One (topology, rho): k_min at e = 0.10, the planned placement, then k_min and planned-placement checks at
    every scenario e."""
    topo_name, rho, total, sla, mus, scen = task
    pcfg = with_params(tuned_cfg(_W["pcfg"], ALG, _W["tuned"]), {"objective.sla_ms": sla})
    ccfg = _W["ccfg"]
    topo, D = topology_and_latency(topo_name, str(_W["paths"].topologies))
    out = []
    kmin, plan = {}, None
    for e in [0.10] + [x for x in scen if x != 0.10]:
        mu = mus[e]
        found = None
        start = 1
        for k in range(start, min(ccfg["k_max"], topo.n - 1) + 1):
            problem, best = _best(topo, D, k, total, mu, pcfg, ccfg)
            d = problem.describe(best.best)
            if feasible(d, k, ccfg):
                found = (k, best.best, d)
                break
        kmin[e] = found[0] if found else None
        if e == 0.10:
            plan = found
        row = {"topology": topo_name, "rho": rho, "e": e, "mu_controller": mu, "k_min": kmin[e]}
        if plan is not None:
            k0, C0, _ = plan
            d0 = build_problem(topo, D, k0, total, mu, pcfg).describe(C0)
            row.update({"planned_k": k0, "planned_feasible": bool(feasible(d0, k0, ccfg)), "planned_pct_sla": d0["pct_sla"],
                        "planned_fail_pct_sla": d0["fail_pct_sla"], "planned_overload": d0["overload"]})
        out.append(row)
    return out


def main():
    p = parser(__doc__, datasets=False)
    args = p.parse_args()
    P = Paths(False)
    od = P.v3 / "escalation_shift"; od.mkdir(parents=True, exist_ok=True)
    fa = od / "scenarios.json"
    if fa.exists():
        rows = json.loads(fa.read_text())
    else:
        rows = part_a(P); save_json(rows, fa)
    df = pd.DataFrame([{"setting": r["setting"], "shift": r["shift"], "seed": r["seed"], "budget": b, **r[f"b{int(100 * b)}"],
                        "f1_student": r["f1_student"], "f1_expert": r["f1_expert"]} for r in rows for b in BUDGETS])
    df.to_csv(od / "scenarios.csv", index=False)
    s10 = df[df.budget == 0.10].groupby("setting")[["e_test", "f1_planned", "f1_capped", "f1_student", "f1_expert"]].median()
    s10.to_csv(od / "scenarios_b10.csv")
    log.info("escalation at a 10%% validation budget (median over seeds):\n%s", s10.round(3).to_string())
    scen = sorted({round(float(x), 2) for x in s10.e_test} | {0.10})
    # Part B
    pcfg, ccfg = load_config("placement", False), load_config("coupled", False)
    entry = load_json(P.outputs / "latency" / "latency.json")["ciciot_class34"]
    mu_s, mu_e = entry["shield"]["throughput_b256"], entry["teacher"]["throughput_b256"]
    mus = {e: controller_mu(pcfg, 1.0 / (1.0 / mu_s + e / mu_e)) for e in scen}
    mu_ref = controller_mu(pcfg, mu_e)
    tuned_f = P.v3 / "placement" / "tuned_params.json"
    tuned = load_json(tuned_f) if tuned_f.exists() else None
    done = {(r["topology"], r["rho"]) for r in read_jsonl(od / "placement.jsonl")}
    tasks = []
    for topo in ccfg["topologies"]:
        t, D = topology_and_latency(topo, str(P.topologies))
        lam = make_demand(t.n, 1.0, pcfg["demand_sigma"], DEMAND_SEED)
        sla = topology_sla(D, lam, ccfg.get("sla", {}), ccfg["k_ref"], pcfg["objective"]["sla_ms"])
        for rho in ccfg["rho_values"]:
            if (topo, float(rho)) not in done:
                tasks.append((topo, float(rho), rho * ccfg["k_ref"] * mus[0.10], sla, mus, scen))
    log.info("scenarios e: %s; %d (topology, rho) tasks", scen, len(tasks))
    with mp.get_context("spawn").Pool(min(len(tasks), ccfg["workers"]) or 1, initializer=_init, initargs=(tuned,)) as pool:
        for res in pool.imap_unordered(_task, tasks):
            for row in res:
                append_jsonl(row, od / "placement.jsonl")
            log.info("done %s rho %s", res[0]["topology"], res[0]["rho"])
    pl = pd.DataFrame(read_jsonl(od / "placement.jsonl"))
    pl.to_csv(od / "placement.csv", index=False)
    save_json({"scenarios_e": scen, "mu_student_flow": mu_s, "mu_expert_flow": mu_e, "mu_controller": mus,
               "algorithm": ALG}, od / "meta.json")
    log.info("placement under escalation shift:\n%s", pl.to_string())


if __name__ == "__main__":
    main()
