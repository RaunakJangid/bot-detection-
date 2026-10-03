"""Validation-only tuning of the distillation objective and per-task feature/size budget
(see src/shield/kd/tuning.py) -> outputs/tuning/kd_choice.json."""

from shield.cli import parser, paths_for, registry
from shield.kd.pool import run_all_specs
from shield.kd.tuning import (ROOT, choose_features, choose_objective, feature_grid, feature_name, mean_val,
                              objective_grid, objective_name)
from shield.utils.config import load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("tune_kd")


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--jobs", type=int, default=None)
    args = p.parse_args()
    paths, reg, kcfg = paths_for(args), registry(args), load_config("kd", args.smoke)
    out_file = paths.out("tuning") / "kd_choice.json"
    if out_file.exists() and not args.force:
        log.info("KD tuning already done: %s", load_json(out_file)["objective"])
        return
    tcfg, jobs = kcfg["tuning"], args.jobs or kcfg["jobs"]
    tasks = [(ds, t) for ds in reg.resolve("main") for t in reg.tasks(ds)]
    key = lambda ds, t: f"{reg.source(ds)}_{t}"
    base = {"seed": tcfg["seed"], "smoke": args.smoke, "cpu": args.cpu, "force": args.force, "root": ROOT,
            "use_tuning": False}

    def evaluate(named_specs: list[tuple[str, dict]], **extra) -> tuple[dict, dict]:
        specs = [{**base, **extra, "dataset": ds, "task": t, "variant": name, "run_name": name, "variant_spec": vs}
                 for name, vs in named_specs for ds, t in tasks]
        results = run_all_specs(specs, jobs)
        scores, params = {}, {}
        for s, m in zip(specs, results):
            scores.setdefault(s["run_name"], {})[key(s["dataset"], s["task"])] = m["best_val_macro_f1"]
            params.setdefault(s["run_name"], {})[key(s["dataset"], s["task"])] = m["params"]
        return scores, params

    # Step 1: objective (all features, no attribution term).
    grid = objective_grid(tcfg)
    scores, _ = evaluate([(objective_name(o), {"select": "all", "gamma": 0.0, **o}) for o in grid])
    obj = choose_objective(scores, grid)
    log.info("Step 1 objective: %s (mean val %.4f)", obj, mean_val(scores, objective_name(obj)))

    # Step 2: same objective, distilled from a teacher assistant.
    asst = {**kcfg["assistant"], **{k: v for k, v in obj.items() if k != "teacher"}, "teacher": "teacher"}
    evaluate([("assistant", asst)], make_cache=True, seed=kcfg["teacher_seed"])
    ta = {**obj, "teacher": "assistant"}
    ta_scores, _ = evaluate([(objective_name(ta), {"select": "all", "gamma": 0.0, **ta})])
    scores.update(ta_scores)
    if mean_val(scores, objective_name(ta)) > mean_val(scores, objective_name(obj)):
        obj = ta
    obj.setdefault("teacher", "teacher")
    log.info("Step 2 teacher: %s", obj["teacher"])

    # Step 3: feature ranking, k and student size with the chosen objective.
    fgrid = feature_grid(tcfg)
    f_scores, f_params = evaluate([(feature_name(c), {"select": "shap", "gamma": 0.0, **obj, **c}) for c in fgrid])
    ranking, per_task = choose_features(f_scores, f_params, fgrid, [key(ds, t) for ds, t in tasks],
                                        tcfg["tolerance"])
    log.info("Step 3 ranking %s, per task %s", ranking, per_task)
    save_json({"objective": obj, "ranking": ranking, "per_task": per_task,
               "objective_scores": scores, "feature_scores": f_scores, "feature_params": f_params,
               "selected_on": "best validation macro-F1, seed %d, main datasets" % tcfg["seed"]}, out_file)


if __name__ == "__main__":
    main()
