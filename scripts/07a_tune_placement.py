"""Tune every placement metaheuristic with the same effort on separate tuning graphs
(src/shield/placement/tuning.py) -> outputs/placement/tuned_params.json, used by 07_placement.py."""

import multiprocessing as mp

import numpy as np
import pandas as pd

from shield.cli import parser, paths_for
from shield.placement.runner import build_problem, controller_mu, detector_mu_flow, run_algorithm, \
    topology_and_latency, with_params
from shield.placement.tuning import candidates, choose, score, tuning_instances
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("tune_placement")
_W: dict = {}


def _init(smoke: bool, mu: float):
    _W["cfg"] = load_config("placement", smoke)
    _W["paths"] = Paths(smoke)
    _W["mu"] = mu


def _run(task: tuple) -> dict:
    alg, ci, params, topo_name, k, run = task
    cfg = _W["cfg"]
    topo, D = topology_and_latency(topo_name, str(_W["paths"].topologies))
    rho = cfg["tuning"]["rho"]
    problem = build_problem(topo, D, k, rho * k * _W["mu"], _W["mu"], cfg)
    res = run_algorithm(alg, problem, topo, with_params(cfg, params), seed=10_000 + run)
    return {"algorithm": alg, "candidate": ci, "topology": topo_name, "k": k, "run": run, "F": res.best_F}


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--v3", action="store_true", help="write to the v3 root (outputs_v3/placement)")
    args = p.parse_args()
    paths, cfg = paths_for(args), load_config("placement", args.smoke)
    out_dir = paths.v3 / "placement" if args.v3 else paths.out("placement")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tuned_params.json"
    if out_file.exists() and not args.force:
        log.info("Placement tuning already done (%s)", out_file)
        return
    tcfg = cfg["tuning"]
    c = cfg["controller"]
    mu_flow, _ = detector_mu_flow(paths.outputs / "latency" / "latency.json", c.get("capacity_run"),
                                  c["capacity_model"], c["default_mu_flow"])
    mu = controller_mu(cfg, mu_flow)
    rng = np.random.default_rng(cfg["seed"])
    cands = {alg: candidates(space, tcfg["candidates"], rng) for alg, space in tcfg["space"].items()}
    insts = tuning_instances(tcfg)
    tasks = [(alg, ci, params, topo, k, r) for alg, cs in cands.items() for ci, params in enumerate(cs)
             for topo, k in insts for r in range(tcfg["runs"])]
    log.info("%d tuning runs: %d algorithms x %s candidates x %d instances x %d runs", len(tasks), len(cands),
             {a: len(c) for a, c in cands.items()}, len(insts), tcfg["runs"])
    with mp.get_context("spawn").Pool(args.workers or cfg["workers"], initializer=_init,
                                      initargs=(args.smoke, mu)) as pool:
        rows = list(pool.imap_unordered(_run, tasks, chunksize=4))
    pd.DataFrame(rows).to_csv(out_dir / "tuning_runs.csv", index=False)
    scores = score(rows)
    scores.to_csv(out_dir / "tuning_scores.csv", index=False)
    tuned = choose(scores, cands, tcfg.get("inherit", {}))
    tuned["_meta"] = {"instances": insts, "runs": tcfg["runs"], "rho": tcfg["rho"], "mu_controller": mu,
                      "note": "tuning graphs use different seeds from every evaluation graph"}
    save_json(tuned, out_file)
    for alg, t in tuned.items():
        if not alg.startswith("_"):
            log.info("%s: score %.4f (default %.4f) params %s", alg, t["score"], t["default_score"] or float("nan"),
                     t["params"])


if __name__ == "__main__":
    main()
