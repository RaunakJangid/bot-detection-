"""Run the whole pipeline in order. `--smoke` runs it on tiny data and checks every expected output."""

import subprocess
import sys
import time
from pathlib import Path

from shield.data.registry import Registry
from shield.utils.config import Paths

HERE = Path(__file__).resolve().parent
STEPS = [
    ["01_ingest.py"], ["02_preprocess.py"], ["02b_leakage_audit.py"], ["03_train_teacher.py"],
    ["03b_xgboost.py"], ["03c_baselines.py"], ["04_shap.py"], ["05_cache_teacher.py"], ["05_distill.py"],
    ["05_distill.py", "--k-sweep"], ["06_latency.py"], ["07_placement.py"], ["07b_coupled.py"],
    ["09_cross_dataset.py"], ["08_make_figures.py"],
]


def expected(paths: Paths) -> list[str]:
    o, d = paths.outputs, paths.data
    datasets = Registry(smoke=paths.smoke).names
    return [
        *[f"{d}/processed/{ds}/meta.json" for ds in datasets],
        f"{o}/leakage/summary.json",
        *[f"{o}/teacher/{ds}_*/metrics.json" for ds in datasets],
        *[f"{o}/kd/{ds}_*/shield_s*/metrics.json" for ds in datasets],
        f"{o}/xgboost/*/metrics.json", f"{o}/baselines/*_dtree/metrics.json", f"{o}/baselines/*_logreg/metrics.json",
        f"{o}/teacher/*/shap/importance.json", f"{o}/teacher/*/cache/done.json", f"{o}/kd/*/shield_k*_s*/metrics.json",
        f"{o}/latency/latency.json", f"{o}/placement/summary.csv", f"{o}/placement/wilcoxon.csv",
        f"{o}/coupled/min_controllers.csv", f"{o}/cross/zero_shot_runs.csv", f"{o}/cross/fewshot_runs.csv",
        f"{o}/cross/unseen_runs.csv", f"{o}/paper/tables/*.csv", f"{o}/paper/figures/*.png",
        f"{o}/paper/figures/cd_kd_ablation.png", f"{o}/paper/figures/cd_placement.png",
        f"{o}/paper/tables/leakage_inflation.csv",
    ]


def main():
    passthrough = sys.argv[1:]
    smoke = "--smoke" in passthrough
    t_all = time.time()
    for step in STEPS:
        cmd = [sys.executable, str(HERE / step[0]), *step[1:], *passthrough]
        print(f"\n=== {' '.join(step)} ===", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"=== {' '.join(step)} done in {time.time() - t0:.0f}s ===", flush=True)
    print(f"\nPipeline finished in {(time.time() - t_all) / 60:.1f} min")
    if smoke:
        paths = Paths(smoke=True)
        missing = []
        for pattern in expected(paths):
            p = Path(pattern)
            anchor = Path(p.anchor)
            if not list(anchor.glob(str(p.relative_to(anchor)))):
                missing.append(pattern)
        if missing:
            print("SMOKE CHECK FAILED, missing outputs:\n  " + "\n  ".join(missing))
            raise SystemExit(1)
        print("Smoke check passed: every expected output exists.")


if __name__ == "__main__":
    main()
