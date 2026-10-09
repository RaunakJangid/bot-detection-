"""v3 step 5: student -> expert cascade (src/shield/eval/cascade.py).

For each main task, student variant, seed and expert ("teacher" = the KD teacher, seed 0; "ensemble" =
the step-1 teacher ensemble): decision rules (step 2) are fitted on validation for student and expert,
then the escalation threshold is set on validation per budget. Test is scored once per budget.
Writes outputs_v3/cascade/<dataset>_<task>/<student>_s<seed>_<expert>.json and cascade/summary.csv.
"""

import numpy as np
import pandas as pd

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed
from shield.eval.cascade import cascade_curve, matching_budget
from shield.eval.metrics import classification_metrics
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.eval.predictions import model_logits
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("cascade")
BUDGETS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def expert_logits(paths, data, expert, device, tm):
    if expert != "teacher":   # "ensemble" (11_ensemble_teacher.py) or "ens4" (19_detector_v4.py)
        f = {s: paths.v3 / "logits" / f"{data.dataset}_{data.task}" / f"{expert}_s0_{s}.npy" for s in ("validation", "test")}
        return {s: np.load(p).astype(np.float32) for s, p in f.items()} if all(p.exists() for p in f.values()) else None
    return model_logits(paths, data, "teacher", 0, ("validation", "test"), device, tm)


def main():
    p = parser(__doc__)
    p.add_argument("--students", nargs="*", default=["shield", "kd_all", "kd_shap"])
    p.add_argument("--experts", nargs="*", default=["teacher", "ensemble"])
    p.add_argument("--no-rules", action="store_true", help="raw argmax decisions (no step-2 rules)")
    args = p.parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    tag = "raw" if args.no_rules else "rules"
    rows = []
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main,full")]:
        for task in reg.tasks(ds, args.task):
            data = load_processed(paths.processed(ds), task, splits=("train", "validation", "test"))
            n, yv, yt = data.n_classes, data.splits["validation"].y, data.splits["test"].y
            prior = np.bincount(data.splits["train"].y, minlength=n) / len(data.splits["train"].y)
            experts = {}
            for e in args.experts:
                z = expert_logits(paths, data, e, device, tcfg["model"])
                if z is None:
                    log.warning("no %s outputs for %s_%s", e, ds, task)
                    continue
                if not args.no_rules:
                    b = fit_decision_rule(z["validation"], yv, prior, device)["bias"]
                    z = {s: apply_bias(v, b) for s, v in z.items()}
                experts[e] = (z, classification_metrics(yv, z["validation"], n)["macro_f1"],
                              classification_metrics(yt, z["test"], n)["macro_f1"])
            for student in args.students:
                for seed in reg.seeds(ds, kcfg["seeds"]):
                    zs = model_logits(paths, data, student, seed, ("validation", "test"), device, tcfg["model"])
                    if zs is None:
                        continue
                    if not args.no_rules:
                        b = fit_decision_rule(zs["validation"], yv, prior, device)["bias"]
                        zs = {s: apply_bias(v, b) for s, v in zs.items()}
                    for e, (ze, e_val, e_test) in experts.items():
                        out = paths.v3 / "cascade" / tag / f"{ds}_{task}" / f"{student}_s{seed}_{e}.json"
                        if out.exists() and not args.force:
                            rows.extend(load_json(out)["rows"])
                            continue
                        curve = cascade_curve(zs["validation"], ze["validation"], yv, zs["test"], ze["test"], yt,
                                              n, BUDGETS)
                        match = matching_budget(curve, e_val)
                        base = {"dataset": ds, "task": task, "student": student, "seed": seed, "expert": e,
                                "expert_val_f1": e_val, "expert_test_f1": e_test}
                        recs = [{**base, **r, "matched": r is match} for r in curve]
                        save_json({"rows": recs}, out)
                        log.info("%s_%s %s s%d -> %s: student %.4f | matched budget %.0f%% (test esc %.1f%%) "
                                 "-> %.4f | expert %.4f", ds, task, student, seed, e, curve[0]["test_macro_f1"],
                                 100 * match["budget"], 100 * match["test_escalated"], match["test_macro_f1"], e_test)
                        rows.extend(recs)
    # The summary covers every run so far (separate invocations add datasets), not only this invocation's.
    (paths.v3 / "cascade" / tag).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r for f in sorted((paths.v3 / "cascade" / tag).glob("*/*.json")) for r in load_json(f)["rows"]])
    df.to_csv(paths.v3 / "cascade" / tag / "summary.csv", index=False)


if __name__ == "__main__":
    main()
