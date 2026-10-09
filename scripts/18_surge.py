"""v3: attack-surge experiment - does detector-driven isolation protect benign switches?

Scenario: a placement is optimised for normal load. A botnet surge then multiplies the traffic of a random
fraction of switches by M. Controllers run the student -> teacher cascade, so a flow costs
1/mu_s + e/mu_t with e the MEASURED escalation rate of benign vs attack flows (CICIoT 34-class, SHIELD, the
validation threshold of the given budget). Strategies, all evaluated on the TRUE surge load:
  S0  static          normal-load placement, switches re-attach (capacitated greedy assignment)
  S1  replace_k       re-optimised k controllers on the measured surge load (volume only, no detector)
  S1+ replace_k+1     the same with one extra controller
  S2  isolate_k+1     the detector flags switches (flow-level recall/FPR from test data, majority vote of
                      the switch's flows); flagged switches go to one dedicated security controller, the
                      other switches get k re-optimised controllers. Missed attack switches stay in the pool.
Metrics (MODEL-based: M/M/1 controllers on shortest-path latencies, not a packet-level emulation):
benign/attacked SLA share and latency, throughput and dropped-flow share (work above u_max*mu is dropped),
controller utilisation/overload/count, attack exposure (attack flows on controllers that also serve benign
switches), MTTD (window of flows at the switch's arrival rate + one measured student batch), time to
mitigate (MTTD + measured re-optimisation time; switch migration not modelled), and a pre-surge baseline.
Detection metrics (precision, recall, F1, FPR) are MEASURED on test flows -> surge/detection.json.
Writes outputs_v3/surge/runs.csv and surge/summary.csv.
"""

import time

import numpy as np
import pandas as pd

from shield.cli import parser
from shield.eval.cascade import confidence
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.placement.objective import Instance, PlacementProblem, assign_capacitated, make_demand
from shield.placement.runner import (DEMAND_SEED, controller_mu, run_algorithm, topology_and_latency, tuned_cfg)
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("surge")


def escalation_rates(paths, budget):
    """(escalation rate of benign flows, of attack flows, flow-level attack recall, FPR) for SHIELD s0 on
    CICIoT 34-class with the decision rule and the validation threshold of `budget`."""
    from shield.data.datasets import load_processed
    data = load_processed(paths.processed("ciciot"), "class34", splits=("train", "validation", "test"))
    lc = paths.v3 / "logits" / "ciciot_class34"
    z = {s: np.load(lc / f"shield_s0_{s}.npy").astype(np.float32) for s in ("validation", "test")}
    prior = np.bincount(data.splits["train"].y, minlength=data.n_classes) / len(data.splits["train"].y)
    b = fit_decision_rule(z["validation"], data.splits["validation"].y, prior)["bias"]
    zv, zt = apply_bias(z["validation"], b), apply_bias(z["test"], b)
    t = np.quantile(confidence(zv), budget)
    benign = data.classes.index("BenignTraffic")
    yt = data.splits["test"].y
    esc, pred = confidence(zt) < t, zt.argmax(1)
    att, flag = yt != benign, pred != benign
    tp, fp, fn = (att & flag).sum(), (~att & flag).sum(), (att & ~flag).sum()
    det = {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
           "fpr": fp / (~att).sum(), "escalation_benign": esc[~att].mean(), "escalation_attack": esc[att].mean(),
           "budget": budget, "note": "flow-level attack-vs-benign, SHIELD s0 cascade student, CICIoT 34-class test"}
    det = {k: (float(v) if not isinstance(v, str) else v) for k, v in det.items()}
    return det["escalation_benign"], det["escalation_attack"], det["recall"], det["fpr"], det


def make_problem(D, lam, mu, k, pcfg):
    o = pcfg["objective"]
    return PlacementProblem(Instance("surge", D, lam, mu, k), o["weights"], pcfg["controller"]["u_max"],
                            o["overload_penalty"], o["sla_ms"], o["n_reference"], seed=DEMAND_SEED)


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--topologies", nargs="*", default=["Geant2012", "Cogentco", "syn200"])
    p.add_argument("--attack-frac", type=float, default=0.1)
    p.add_argument("--multipliers", type=float, nargs="*", default=[5.0, 10.0, 20.0])
    p.add_argument("--rho", type=float, default=0.7, help="normal-load utilisation of the k controllers")
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--algorithm", default="gwo_sa")
    p.add_argument("--evals", type=int, default=5000)
    p.add_argument("--window", type=int, default=100, help="flows per switch observed before deciding (MTTD)")
    args = p.parse_args()
    paths = Paths(args.smoke)
    pcfg, ccfg = load_config("placement", args.smoke), load_config("coupled", args.smoke)
    tuned = load_json(paths.v3 / "placement" / "tuned_params.json")
    acfg = tuned_cfg(pcfg, args.algorithm, tuned)
    lat_entry = load_json(paths.outputs / "latency" / "latency.json")["ciciot_class34"]
    mu_s, mu_t = lat_entry["shield"]["throughput_b256"], lat_entry["teacher"]["throughput_b256"]
    e_b, e_a, recall, fpr, detection = escalation_rates(paths, args.budget)
    log.info("escalation benign %.3f attack %.3f | flow recall %.3f FPR %.4f", e_b, e_a, recall, fpr)
    mu = controller_mu(pcfg, mu_s)              # controller rate in student-flow units
    cost_b, cost_a = 1 + e_b * mu_s / mu_t, 1 + e_a * mu_s / mu_t   # student-flow units per flow
    sla_meta = load_json(paths.v3 / "coupled" / "coupled_meta.json")["sla_ms"]
    mink = pd.read_csv(paths.v3 / "coupled" / "min_controllers.csv")
    u_max = pcfg["controller"]["u_max"]
    infer_ms = lat_entry["shield"]["latency_b256_ms"]     # one batch of the student (measured)

    def evaluate(D, pos, ctrl, load, benign_f, attack_f, A, sla):
        """Model-based network metrics for switch -> controller mapping `ctrl` (controller node ids `pos`).
        Work above a controller's stable capacity u_max*mu is counted as dropped flows."""
        kk = len(pos)
        W = np.bincount(ctrl, weights=load, minlength=kk)
        Fl = np.bincount(ctrl, weights=benign_f + attack_f, minlength=kk)
        Fa = np.bincount(ctrl, weights=attack_f, minlength=kk)
        served = np.minimum(1.0, u_max * mu / np.maximum(W, 1e-12))
        lat = D[np.arange(len(ctrl)), pos[ctrl]] + 1000.0 / np.maximum(mu - np.minimum(W, u_max * mu), 1e-9)[ctrl]
        shared = np.bincount(ctrl[~A], minlength=kk) > 0          # controllers serving at least one benign switch
        return {"benign_sla": float((lat[~A] <= sla).mean()), "benign_ms": float(lat[~A].mean()),
                "attacked_sla": float((lat[A] <= sla).mean()) if A.any() else np.nan,
                "throughput_flows": float((Fl * served).sum()), "drop_frac": float(1 - (Fl * served).sum() / Fl.sum()),
                # dropping attack flows is mitigation, not damage: the benign drop rate is what users feel
                "benign_drop": float(1 - ((Fl - Fa) * served).sum() / max((Fl - Fa).sum(), 1e-12)),
                "benign_throughput": float(((Fl - Fa) * served).sum()),
                "util_max": float(W.max() / mu), "util_mean": float(W.mean() / mu),
                "overload": float(np.maximum(W / mu - u_max, 0).sum()),
                "attack_exposure": float(Fa[shared].sum() / max(Fa.sum(), 1e-12)) if A.any() else 0.0,
                "k_total": kk}

    def mapping(D, C, load, idx=None):
        """Capacitated greedy assignment of switches (optionally the subset idx) to controllers C."""
        idx = np.arange(len(D)) if idx is None else idx
        prim, _ = assign_capacitated(np.ascontiguousarray(D[np.ix_(idx, C)]), load[idx], u_max * mu)
        return prim

    rows = []
    for topo_name in args.topologies:
        topo, D = topology_and_latency(topo_name, str(paths.topologies))
        sla = sla_meta[topo_name]
        k = int(mink[(mink.topology == topo_name) & (mink.rho == 0.5)]["shield"].iloc[0])
        cfg = {**acfg, "objective": {**acfg["objective"], "sla_ms": sla}}
        lam0 = make_demand(topo.n, args.rho * k * mu / cost_b, pcfg["demand_sigma"], DEMAND_SEED)   # benign flows/s
        P0 = run_algorithm(args.algorithm, make_problem(D, lam0 * cost_b, mu, k, cfg), topo, cfg, 0,
                           budget=args.evals).best
        none = np.zeros(topo.n, bool)
        base = evaluate(D, P0, mapping(D, P0, lam0 * cost_b), lam0 * cost_b, lam0, 0 * lam0, none, sla)
        rows.append({"topology": topo_name, "M": 0.0, "trial": -1, "strategy": "pre_surge", **base})
        for M in args.multipliers:
            for trial in range(args.trials):
                rng = np.random.default_rng(1000 * trial + int(M))
                A = np.zeros(topo.n, bool)
                A[rng.choice(topo.n, max(1, int(round(args.attack_frac * topo.n))), replace=False)] = True
                att = lam0 * A * M                                 # attack flows/s at attacked switches
                load = lam0 * cost_b + att * cost_a                # true surge work (student-flow units)
                # detector: a switch is flagged when most of its flows are predicted as attacks
                share = (att * recall + lam0 * fpr) / (att + lam0)
                flagged = share > 0.5
                # MTTD (model): observe a window of `window` flows at the attacked switch, plus one student batch
                mttd_ms = float(np.median(1000.0 * args.window / (lam0[A] * (1 + M))) + infer_ms)
                common = {"topology": topo_name, "M": M, "trial": trial, "mttd_ms": mttd_ms,
                          "flagged": int(flagged.sum()), "attacked": int(A.sum()),
                          "missed": int((A & ~flagged).sum()), "false_flags": int((~A & flagged).sum())}
                strategies = {"S0_static": (P0, mapping(D, P0, load), 0.0)}
                for name, kk in (("S1_replace_k", k), ("S1_replace_k+1", k + 1)):
                    t0 = time.time()
                    C = run_algorithm(args.algorithm, make_problem(D, load, mu, kk, cfg), topo, cfg, trial,
                                      budget=args.evals).best
                    strategies[name] = (C, mapping(D, C, load), time.time() - t0)
                # S2: flagged switches -> one dedicated security controller (1-median of the flagged switches);
                # the others -> k controllers re-optimised on their own load. Missed attack switches stay in the pool.
                t0 = time.time()
                pool = np.flatnonzero(~flagged)
                Dp = np.ascontiguousarray(D[np.ix_(pool, pool)])
                Cp = pool[run_algorithm(args.algorithm, make_problem(Dp, load[pool], mu, k, cfg), topo, cfg, trial,
                                        budget=args.evals).best]
                ctrl = np.empty(topo.n, np.int64)
                ctrl[pool] = mapping(D, Cp, load, pool)
                pos = Cp
                if flagged.any():
                    fl = np.flatnonzero(flagged)
                    sec = fl[int(np.argmin((load[fl][:, None] * D[np.ix_(fl, fl)]).sum(0)))]
                    pos = np.append(Cp, sec)
                    ctrl[fl] = len(Cp)
                strategies["S2_isolate_k+1"] = (pos, ctrl, time.time() - t0)
                for name, (pos, ctrl, secs) in strategies.items():
                    m = evaluate(D, np.asarray(pos), ctrl, load, lam0, att, A, sla)
                    rows.append({**common, "strategy": name, **m, "replace_seconds": secs,
                                 "time_to_mitigate_s": (mttd_ms / 1000 + secs) if name != "S0_static" else np.nan})
            log.info("%s M=%.0f done", topo_name, M)
    df = pd.DataFrame(rows)
    out = paths.v3 / "surge"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    save_json(detection, out / "detection.json")
    cols = ["benign_sla", "benign_ms", "attacked_sla", "throughput_flows", "drop_frac", "benign_drop", "util_max", "overload",
            "attack_exposure", "k_total", "mttd_ms", "time_to_mitigate_s"]
    summ = df.groupby(["topology", "M", "strategy"])[cols].mean()
    summ.to_csv(out / "summary.csv")
    log.info("\n%s", summ.round(3).to_string())


if __name__ == "__main__":
    main()
