"""Controller placement benchmark: every algorithm x topology x (k, rho) x seed, with resilience analysis.

Resumable: finished runs are appended to outputs/placement/runs.jsonl and skipped on restart.
"""

import multiprocessing as mp
import time
from functools import lru_cache

import pandas as pd

from shield.cli import parser, paths_for
from shield.eval.resilience import betweenness, resilience_report
from shield.eval.stats import friedman_ranks, summarise, wilcoxon_vs
from shield.placement.exact import brute_force
from shield.placement.runner import (DETERMINISTIC, STOCHASTIC, build_problem, compress_history,
                                     controller_mu, detector_mu_flow, run_algorithm, topology_and_latency)
from shield.placement.topology import download_zoo
from shield.utils.config import Paths, load_config
from shield.utils.io import append_jsonl, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("placement")
_W: dict = {}


def instances(cfg: dict) -> list[tuple[str, int, float]]:
    topologies = list(cfg["zoo"]) + [f"syn{n}" for n in cfg["synthetic"]]
    out = []
    for topo in topologies:
        for k in cfg["k_values"]:
            sweep = cfg["rho_sweep_k"] is None or k == cfg["rho_sweep_k"]
            for rho in (cfg["rho_values"] if sweep else [cfg["rho_default"]]):
                out.append((topo, k, float(rho)))
    return out


def _init(smoke: bool, mu: float):
    _W["cfg"] = load_config("placement", smoke)
    _W["paths"] = Paths(smoke)
    _W["mu"] = mu


@lru_cache(maxsize=64)
def _problem(topo_name: str, k: int, rho: float):
    topo, D = topology_and_latency(topo_name, str(_W["paths"].topologies))
    return topo, build_problem(topo, D, k, rho * k * _W["mu"], _W["mu"], _W["cfg"])


@lru_cache(maxsize=16)
def _bc(topo_name: str):
    topo, _ = topology_and_latency(topo_name, str(_W["paths"].topologies))
    return betweenness(topo.G)


def _run(task: tuple) -> dict:
    topo_name, k, rho, alg, seed = task
    cfg = _W["cfg"]
    topo, problem = _problem(topo_name, k, rho)
    base = {"topology": topo_name, "k": k, "rho": rho, "algorithm": alg, "seed": seed, "n": topo.n}
    if alg == "__optimum__":
        r = brute_force(problem, cfg["brute_force_limit"])
        return {**base, "skipped": r is None, **({} if r is None else {
            "F_opt": r["best_F"], "best": r["best"], "evaluated": r["evaluated"], "seconds": r["seconds"]})}
    res = run_algorithm(alg, problem, topo, cfg, seed)
    if res is None:
        return {**base, "skipped": True}
    row = {**base, "skipped": False, **problem.describe(res.best), "best": res.best, "n_evals": res.n_evals,
           "seconds": res.seconds, "history": compress_history(res.history, cfg["budget"])}
    edge_bc, node_bc = _bc(topo_name)
    rep = resilience_report(topo.G, problem, res.best, cfg["resilience"], seed, edge_bc, node_bc)
    _flatten("res", rep, row)
    return row


def _flatten(prefix: str, d: dict, row: dict) -> None:
    """{'controller': {'single': {'worst_avg_ms': x}}} -> row['res_controller_single_worst_avg_ms'] = x"""
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(f"{prefix}_{k}", v, row)
        else:
            row[f"{prefix}_{k}"] = v


def _key(row_or_task) -> tuple:
    if isinstance(row_or_task, dict):
        r = row_or_task
        return (r["topology"], int(r["k"]), float(r["rho"]), r["algorithm"], int(r["seed"]))
    return tuple(row_or_task)


def analyse(out_dir, cfg) -> None:
    rows = [r for r in read_jsonl(out_dir / "runs.jsonl") if not r.get("skipped")]
    if not rows:
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "best")} for r in rows])
    df.to_parquet(out_dir / "runs.parquet")
    opt_rows = [r for r in read_jsonl(out_dir / "optimum.jsonl") if not r.get("skipped")]
    opt = pd.DataFrame(opt_rows) if opt_rows else None
    summary = summarise(df, opt)
    summary.to_csv(out_dir / "summary.csv", index=False)
    others = [a for a in cfg["algorithms"] if a in STOCHASTIC and a != "hybrid_eho_aco"]
    wilcoxon_vs(df, "hybrid_eho_aco", others).to_csv(out_dir / "wilcoxon.csv", index=False)
    ranks, p = friedman_ranks(summary, cfg["algorithms"])
    save_json({"average_rank": ranks.to_dict(), "friedman_p": p}, out_dir / "friedman.json")
    log.info("Average ranks (lower is better):\n%s", ranks.to_string())


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--analyse-only", action="store_true")
    args = p.parse_args()
    paths, cfg = paths_for(args), load_config("placement", args.smoke)
    out_dir = paths.out("placement")
    if args.analyse_only:
        analyse(out_dir, cfg)
        return
    for name in cfg["zoo"]:
        download_zoo(name, paths.topologies)

    c = cfg["controller"]
    mu_flow, source = detector_mu_flow(paths.outputs / "latency" / "latency.json", c.get("capacity_run"),
                                       c["capacity_model"], c["default_mu_flow"])
    mu = controller_mu(cfg, mu_flow)
    save_json({"mu_flow": mu_flow, "mu_controller": mu, "source": source}, out_dir / "capacity.json")
    log.info("Detector throughput %.0f flows/s/core (%s) -> controller mu %.0f flows/s", mu_flow, source, mu)

    if args.force:
        for f in ("runs.jsonl", "optimum.jsonl"):
            (out_dir / f).unlink(missing_ok=True)
    done = {_key(r) for r in read_jsonl(out_dir / "runs.jsonl")}
    done_opt = {_key(r) for r in read_jsonl(out_dir / "optimum.jsonl")}
    tasks, opt_tasks = [], []
    for topo, k, rho in instances(cfg):
        for alg in cfg["algorithms"]:
            for seed in (range(cfg["runs"]) if alg in STOCHASTIC else [0]):
                t = (topo, k, rho, alg, seed)
                if t not in done:
                    tasks.append(t)
        t = (topo, k, rho, "__optimum__", 0)
        if t not in done_opt:
            opt_tasks.append(t)
    unknown = set(cfg["algorithms"]) - set(STOCHASTIC) - set(DETERMINISTIC)
    if unknown:
        raise SystemExit(f"Unknown algorithms in config: {sorted(unknown)}")
    # Longest jobs first (big graphs, ILP) for better load balance.
    weight = {"ilp_ckm": 50.0}
    tasks.sort(key=lambda t: -topology_and_latency(t[0], str(paths.topologies))[0].n * t[1] * weight.get(t[3], 1.0))
    log.info("%d runs + %d optimum checks to do (%d already done)", len(tasks), len(opt_tasks), len(done))

    workers = args.workers or cfg["workers"]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(args.smoke, mu)) as pool:
        for i, row in enumerate(pool.imap_unordered(_run, opt_tasks), 1):
            append_jsonl(row, out_dir / "optimum.jsonl")
        for i, row in enumerate(pool.imap_unordered(_run, tasks), 1):
            append_jsonl(row, out_dir / "runs.jsonl")
            if i % 100 == 0 or i == len(tasks):
                rate = i / (time.time() - t0)
                log.info("%d/%d runs (%.1f/s, ~%.0f min left)", i, len(tasks), rate,
                         (len(tasks) - i) / max(rate, 1e-9) / 60)
    analyse(out_dir, cfg)


if __name__ == "__main__":
    main()
