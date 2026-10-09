"""Run many distillation specs, `jobs` at a time on the GPU (spawned processes)."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from shield.kd.trainer import run_distillation
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.logging import get_logger

log = get_logger("distill")


def run_spec(spec: dict) -> dict:
    """spec: dataset, task, variant, seed, smoke, cpu, force, and optionally k, variant_spec, run_name,
    root, use_tuning, make_cache."""
    paths = Paths(spec["smoke"])
    kcfg, tcfg = load_config("kd", spec["smoke"]), load_config("teacher", spec["smoke"])
    device = get_device(spec["cpu"], spec["smoke"])
    return run_distillation(paths, spec["dataset"], spec["task"], spec["variant"], spec["seed"], kcfg,
                            tcfg["model"], device, spec["force"], k_override=spec.get("k"),
                            variant_spec=spec.get("variant_spec"), run_name=spec.get("run_name"),
                            root=spec.get("root", "kd"), use_tuning=spec.get("use_tuning"),
                            make_cache=spec.get("make_cache", False), v3=spec.get("v3", False))


def _label(spec: dict, m: dict) -> str:
    return f"{spec['dataset']}_{spec['task']}/{m['run']}_s{spec['seed']}: macro-F1 {m['test']['macro_f1']:.4f}"


def run_all_specs(specs: list[dict], jobs: int) -> list[dict]:
    """Returns the metrics of every spec (in input order). Raises if any run failed."""
    log.info("%d distillation runs, %d at a time", len(specs), jobs)
    results: list[dict | None] = [None] * len(specs)
    if jobs <= 1:
        for i, spec in enumerate(specs):
            results[i] = run_spec(spec)
            log.info(_label(spec, results[i]))
        return results
    failures = 0
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(run_spec, s): i for i, s in enumerate(specs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
                log.info(_label(specs[i], results[i]))
            except Exception as exc:
                failures += 1
                log.error("FAILED %s: %r", specs[i], exc)
    if failures:
        raise SystemExit(f"{failures} distillation runs failed")
    return results
