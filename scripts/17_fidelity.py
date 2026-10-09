"""v3: does the attribution loss make the student's explanations more faithful to the teacher's?

v2 measured student-vs-teacher SHAP agreement only for seed 0. This computes it for every seed of the
variants that share features and size (shield vs kd_shap: they differ only in the attribution loss),
with the same procedure (kd.trainer.student_fidelity), and a paired Wilcoxon over (task, seed) pairs.
Writes outputs_v3/fidelity/<dataset>_<task>/<variant>_s<seed>.json, fidelity/summary.csv, fidelity/wilcoxon.json.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed
from shield.eval.predictions import load_student, run_dir
from shield.kd.trainer import student_fidelity
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.xai.shap_analysis import shap_dir

log = get_logger("fidelity")


def main():
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=["shield", "kd_shap"])
    args = p.parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    kcfg, tm = load_config("kd", args.smoke), load_config("teacher", args.smoke)["model"]
    rows = []
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main,full")]:
        for task in reg.tasks(ds, args.task):
            data = load_processed(paths.processed(ds), task, splits=("train",))
            imp = np.asarray(load_json(shap_dir(paths, ds, task, tm, kcfg["teacher_seed"]) / "importance.json")["importance"])
            for v in args.variants:
                for seed in reg.seeds(ds, kcfg["seeds"]):
                    out = paths.v3 / "fidelity" / f"{ds}_{task}" / f"{v}_s{seed}.json"
                    if out.exists() and not args.force:
                        rows.append(load_json(out))
                        continue
                    model, S = load_student(run_dir(paths, ds, task, v, seed, tm), device)
                    fid = student_fidelity(paths, ds, task, tm, kcfg["teacher_seed"], model, S, data.splits["train"].X,
                                           imp, kcfg["fidelity"], device, seed)
                    row = {"dataset": ds, "task": task, "variant": v, "seed": seed, "spearman": fid["spearman"],
                           **{k: fid[k] for k in fid if k.startswith("top")}}
                    save_json(row, out)
                    log.info("%s_%s %s s%d: spearman %.3f", ds, task, v, seed, row["spearman"])
                    rows.append(row)
    df = pd.DataFrame(rows)
    (paths.v3 / "fidelity").mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.v3 / "fidelity" / "summary.csv", index=False)
    a, b = args.variants[:2]
    piv = df.pivot_table(index=["dataset", "task", "seed"], columns="variant", values="spearman").dropna()
    d = (piv[a] - piv[b]).to_numpy()
    res = {"pairs": len(d), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
           "p_two_sided": float(wilcoxon(d).pvalue) if len(d) > 1 else None,
           "per_task": piv.groupby(level=[0, 1]).mean().round(3).reset_index().to_dict("records")}
    save_json(res, paths.v3 / "fidelity" / "wilcoxon.json")
    log.info("%s vs %s explanation agreement: mean diff %.3f, %d/%d wins, p = %s", a, b, res["mean_diff"],
             res["wins"], res["pairs"], res["p_two_sided"])


if __name__ == "__main__":
    main()
