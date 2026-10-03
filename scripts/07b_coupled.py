"""Coupling experiment: with demand fixed, how many controllers does each detector need to meet the SLA?

A faster detector raises every controller's service rate mu_c, so fewer controllers carry the same load.
The SLA is set per topology (coupled.yaml `sla`), and placements use the tuned hybrid EHO-ACO.
"""

import multiprocessing as mp

import pandas as pd

from shield.cli import parser, paths_for
from shield.placement.coupling import feasible, topology_sla
from shield.placement.objective import make_demand
from shield.placement.runner import (DEMAND_SEED, build_problem, controller_mu, run_algorithm, topology_and_latency,
                                     tuned_cfg, with_params)
from shield.placement.topology import download_zoo
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, load_json, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("coupled")
_W: dict = {}


def _init(smoke: bool, tuned: dict | None):
    _W["pcfg"] = load_config("placement", smoke)
    _W["ccfg"] = load_config("coupled", smoke)
    _W["paths"] = Paths(smoke)
    _W["tuned"] = tuned


def _run(task: tuple) -> dict:
    topo_name, rho, model, k, mu, total, sla = task
    pcfg = with_params(tuned_cfg(_W["pcfg"], "hybrid_eho_aco", _W["tuned"]), {"objective.sla_ms": sla})
    ccfg = _W["ccfg"]
    topo, D = topology_and_latency(topo_name, str(_W["paths"].topologies))
    base = {"topology": topo_name, "rho": rho, "model": model, "k": k, "mu_controller": mu, "total_demand": total,
            "sla_ms": sla}
    if k >= topo.n:
        return {**base, "skipped": True}
    problem = build_problem(topo, D, k, total, mu, pcfg)
    best = None
    for seed in range(ccfg["runs"]):
        res = run_algorithm("hybrid_eho_aco", problem, topo, pcfg, seed, budget=ccfg["budget"])
        if best is None or res.best_F < best.best_F:
            best = res
    d = problem.describe(best.best)
    return {**base, "skipped": False, **d, "feasible": feasible(d, k, ccfg), "best": best.best}


def main():
    args = parser(__doc__, datasets=False).parse_args()
    paths = paths_for(args)
    pcfg, ccfg = load_config("placement", args.smoke), load_config("coupled", args.smoke)
    out_dir = paths.out("coupled")
    lat_file = paths.outputs / "latency" / "latency.json"
    if not lat_file.exists():
        raise SystemExit(f"{lat_file} not found; run 06_latency.py first")
    table = load_json(lat_file)
    run_key = pcfg["controller"]["capacity_run"]
    if run_key not in table:
        run_key = next(k for k in table if not k.startswith("_"))
    entry = table[run_key]
    missing = [m for m in ccfg["models"] + [ccfg["demand_reference_model"]] if m not in entry]
    if missing:
        raise SystemExit(f"latency.json[{run_key}] lacks {missing}")
    mu = {m: controller_mu(pcfg, entry[m]["throughput_b256"]) for m in set(ccfg["models"]) | {ccfg["demand_reference_model"]}}
    for name in ccfg["topologies"]:
        if not name.startswith("syn"):
            download_zoo(name, paths.topologies)
    tuned_file = paths.outputs / "placement" / "tuned_params.json"
    tuned = load_json(tuned_file) if tuned_file.exists() else None

    sla = {}
    for topo in ccfg["topologies"]:
        t, D = topology_and_latency(topo, str(paths.topologies))
        lam = make_demand(t.n, 1.0, pcfg["demand_sigma"], DEMAND_SEED)  # only the shape matters for k-median
        sla[topo] = topology_sla(D, lam, ccfg.get("sla", {}), ccfg["k_ref"], pcfg["objective"]["sla_ms"])
    log.info("SLA per topology (ms): %s", {k: round(v, 2) for k, v in sla.items()})

    if args.force:
        (out_dir / "coupled.jsonl").unlink(missing_ok=True)
    done = {(r["topology"], r["rho"], r["model"], r["k"]) for r in read_jsonl(out_dir / "coupled.jsonl")}
    tasks = []
    for topo in ccfg["topologies"]:
        for rho in ccfg["rho_values"]:
            total = rho * ccfg["k_ref"] * mu[ccfg["demand_reference_model"]]
            for model in ccfg["models"]:
                for k in range(1, ccfg["k_max"] + 1):
                    if (topo, float(rho), model, k) not in done:
                        tasks.append((topo, float(rho), model, k, mu[model], total, sla[topo]))
    log.info("Capacity source %s; mu_c: %s; %d tasks", run_key, {m: round(v) for m, v in mu.items()}, len(tasks))
    with mp.get_context("spawn").Pool(ccfg["workers"], initializer=_init, initargs=(args.smoke, tuned)) as pool:
        for row in pool.imap_unordered(_run, tasks):
            append_jsonl(row, out_dir / "coupled.jsonl")

    df = pd.DataFrame([r for r in read_jsonl(out_dir / "coupled.jsonl") if not r.get("skipped")])
    df.drop(columns=["best"]).to_csv(out_dir / "coupled_all.csv", index=False)
    # Every (topology, rho) appears; blank = no k <= k_max met the SLA.
    full = pd.MultiIndex.from_product([ccfg["topologies"], [float(r) for r in ccfg["rho_values"]]],
                                      names=["topology", "rho"])
    mink = (df[df.feasible].groupby(["topology", "rho", "model"])["k"].min()
            .unstack("model").reindex(index=full, columns=ccfg["models"]))
    mink.to_csv(out_dir / "min_controllers.csv")
    save_json({"run_key": run_key, "mu_controller": mu, "k_max": ccfg["k_max"], "sla_ms": sla,
               "sla_rule": ccfg.get("sla"), "tuned_hybrid": bool(tuned and "hybrid_eho_aco" in tuned)},
              out_dir / "coupled_meta.json")
    log.info("Minimum controllers meeting the SLA (blank = more than k_max):\n%s", mink.to_string())


if __name__ == "__main__":
    main()
