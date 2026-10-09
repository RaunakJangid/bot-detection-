"""v3 step 1 (students): distil from the teacher ensemble (11_ensemble_teacher.py) instead of the single
ResMLP teacher, with everything else identical to v2 (validation-tuned objective, per-task k and size,
same seeds). gamma = 0: the ensemble has no input gradients, and v2 found the attribution loss gives no gain.

    kd_all_ens   all features   (compare with v2 kd_all)
    kd_shap_ens  SHAP top-k     (compare with v2 kd_shap / shield)
Runs go to outputs_v3/kd/<dataset>_<task>/<variant>_s<seed>.
"""

from shield.cli import parser, registry, selected_datasets
from shield.kd.pool import run_all_specs
from shield.utils.config import load_config

VARIANTS = {
    "kd_all_ens": {"select": "all", "gamma": 0.0, "teacher": "ensemble"},
    "kd_shap_ens": {"select": "shap", "gamma": 0.0, "teacher": "ensemble"},
}


def main():
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=None)
    args = p.parse_args()
    reg, kcfg = registry(args), load_config("kd", args.smoke)
    specs = [{"smoke": args.smoke, "cpu": args.cpu, "force": args.force, "dataset": ds, "task": task,
              "variant": v, "run_name": v, "variant_spec": VARIANTS[v], "seed": s, "v3": True}
             for ds in selected_datasets(args) if ds in reg.resolve("main,full")
             for task in reg.tasks(ds, args.task)
             for v in args.variants
             for s in (args.seeds if args.seeds is not None else reg.seeds(ds, kcfg["seeds"]))]
    run_all_specs(specs, args.jobs or kcfg["jobs"])


if __name__ == "__main__":
    main()
