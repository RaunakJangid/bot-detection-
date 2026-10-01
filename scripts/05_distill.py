"""Distillation ablation grid: every (dataset, task, variant, seed); `jobs` runs share the GPU."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from shield.cli import parser, paths_for, selected_datasets, tasks_for
from shield.kd.trainer import run_distillation
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.logging import get_logger

log = get_logger("distill")


def _run(spec: dict) -> str:
    paths = Paths(spec["smoke"])
    kcfg, tcfg = load_config("kd", spec["smoke"]), load_config("teacher", spec["smoke"])
    scfg = load_config("shap", spec["smoke"])
    device = get_device(spec["cpu"], spec["smoke"])
    m = run_distillation(paths, spec["dataset"], spec["task"], spec["variant"], spec["seed"], kcfg,
                         tcfg["model"], scfg, device, spec["force"], k_override=spec["k"])
    return f"{spec['dataset']}_{spec['task']}/{m['run']}_s{spec['seed']}: macro-F1 {m['test']['macro_f1']:.4f}"


def main():
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--k-sweep", action="store_true",
                   help="instead of the ablation grid, run the given variants (default: shield) at every k in shap.yaml k_values")
    args = p.parse_args()
    paths_for(args)
    kcfg, tcfg = load_config("kd", args.smoke), load_config("teacher", args.smoke)
    if args.k_sweep:
        variants = args.variants or ["shield"]
        ks = load_config("shap", args.smoke)["k_values"]
    else:
        variants = args.variants or list(kcfg["variants"])
        ks = [None]
    specs = [
        {"dataset": ds, "task": task, "variant": v, "seed": s, "k": k, "smoke": args.smoke, "cpu": args.cpu,
         "force": args.force}
        for ds in selected_datasets(args)
        for task in tasks_for(tcfg["tasks"], ds, args.task)
        for v in variants
        for k in ks
        for s in (args.seeds if args.seeds is not None else kcfg["seeds"])
    ]
    jobs = args.jobs or kcfg["jobs"]
    log.info("%d distillation runs, %d at a time", len(specs), jobs)
    if jobs <= 1:
        for spec in specs:
            log.info(_run(spec))
        return
    failures = 0
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(_run, s): s for s in specs}
        for fut in as_completed(futs):
            try:
                log.info(fut.result())
            except Exception as exc:
                failures += 1
                log.error("FAILED %s: %r", futs[fut], exc)
    if failures:
        raise SystemExit(f"{failures} distillation runs failed")


if __name__ == "__main__":
    main()
