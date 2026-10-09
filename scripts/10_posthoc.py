"""v3 step 2: macro-F1 decision rules fitted on validation (logit adjustment + per-class bias).

For every finished v2 model (teacher, XGBoost, each KD variant, every seed) on the main datasets:
fit the rule on validation, then score test once with raw / logit-adjusted / adjusted+bias decisions.
Writes outputs_v3/posthoc/<dataset>_<task>/<model>_s<seed>.json, posthoc/summary.csv and the paired
Wilcoxon of raw vs adjusted test macro-F1 (posthoc/wilcoxon.csv).
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed
from shield.eval.metrics import classification_metrics
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.eval.predictions import model_logits
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("posthoc")
MODELS = ["teacher", "xgboost", "scratch_all", "kd_all", "kd_shap", "shield", "kd_random", "kd_mi", "kd_variance"]


def main():
    p = parser(__doc__)
    p.add_argument("--models", nargs="*", default=MODELS)
    args = p.parse_args()
    paths, reg = Paths(args.smoke), registry(args)
    device = get_device(args.cpu)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    datasets = [d for d in selected_datasets(args) if d in reg.resolve("main,full")]
    rows = []
    for ds in datasets:
        for task in reg.tasks(ds, args.task):
            data = load_processed(paths.processed(ds), task)
            prior = np.bincount(data.splits["train"].y, minlength=data.n_classes).astype(np.float64)
            prior /= prior.sum()
            yv, yt = data.splits["validation"].y, data.splits["test"].y
            for model in args.models:
                seeds = tcfg["xgboost"]["seeds"] if model == "xgboost" else reg.seeds(ds, kcfg["seeds"])
                for seed in seeds:
                    out = paths.v3 / "posthoc" / f"{ds}_{task}" / f"{model}_s{seed}.json"
                    if out.exists() and not args.force:
                        rows.append(load_json(out)["row"])
                        continue
                    z = model_logits(paths, data, model, seed, ("validation", "test"), device, tcfg["model"])
                    if z is None:
                        log.warning("no run for %s %s %s s%d", ds, task, model, seed)
                        continue
                    rule = fit_decision_rule(z["validation"], yv, prior, device)
                    test = {name: classification_metrics(yt, apply_bias(z["test"], b), data.n_classes,
                                                         data.classes, full=True)
                            for name, b in (("raw", np.zeros(data.n_classes)), ("la", rule["la_bias"]),
                                            ("bias", rule["bias"]))}
                    row = {"dataset": ds, "task": task, "model": model, "seed": seed, "tau": rule["tau"],
                           **{f"val_{k}": rule[f"val_macro_f1_{k}"] for k in ("raw", "la", "bias")},
                           **{f"test_{k}": test[k]["macro_f1"] for k in test},
                           **{f"test_acc_{k}": test[k]["accuracy"] for k in test}}
                    save_json({"row": row, "rule": rule, "test": test}, out)
                    log.info("%s/%s %s s%d: test macro-F1 raw %.4f -> LA %.4f (tau %.2f) -> +bias %.4f",
                             ds, task, model, seed, row["test_raw"], row["test_la"], rule["tau"], row["test_bias"])
                    rows.append(row)
    # Summaries cover every model run so far (separate invocations add models), not only this invocation's.
    df = pd.DataFrame([load_json(f)["row"] for f in sorted((paths.v3 / "posthoc").glob("*/*.json"))])
    df.to_csv(paths.v3 / "posthoc" / "summary.csv", index=False)

    tests = []
    for model, g in df.groupby("model"):
        for rule in ("la", "bias"):
            d = (g[f"test_{rule}"] - g["test_raw"]).to_numpy()
            p_two = float(wilcoxon(d, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
            tests.append({"model": model, "rule": rule, "pairs": len(d), "mean_gain_pts": 100 * d.mean(),
                          "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p_two_sided": p_two})
    pd.DataFrame(tests).to_csv(paths.v3 / "posthoc" / "wilcoxon.csv", index=False)
    log.info("\n%s", pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
