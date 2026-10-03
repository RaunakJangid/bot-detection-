"""Distillation ablation grid: every (dataset, task, variant, seed); `jobs` runs share the GPU.

Uses the validation-tuned objective and per-task feature budget (outputs/tuning/kd_choice.json) when
present. If tuning chose a teacher assistant, the assistants are trained and cached first.
  --k-sweep    run the given variants (default: shield) at every k in shap.yaml k_values
  --low-data   run kd.yaml lowdata.variants on lowdata.fractions of the training split (main datasets)
"""

from shield.cli import parser, registry, selected_datasets
from shield.kd.pool import run_all_specs
from shield.kd.trainer import tuning_choice
from shield.utils.config import Paths, load_config
from shield.utils.logging import get_logger

log = get_logger("distill")


def main():
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--k-sweep", action="store_true")
    mode.add_argument("--low-data", action="store_true")
    args = p.parse_args()
    reg, kcfg = registry(args), load_config("kd", args.smoke)
    common = {"smoke": args.smoke, "cpu": args.cpu, "force": args.force}
    seeds_of = lambda ds: args.seeds if args.seeds is not None else reg.seeds(ds, kcfg["seeds"])

    if args.k_sweep:
        datasets = selected_datasets(args, stage="k_sweep")
        runs = [(v, None, k) for v in (args.variants or ["shield"]) for k in load_config("shap", args.smoke)["k_values"]]
    elif args.low_data:
        datasets = [d for d in selected_datasets(args) if d in reg.resolve("main")]
        runs = [(v, f, None) for v in (args.variants or kcfg["lowdata"]["variants"]) for f in kcfg["lowdata"]["fractions"]]
    else:
        datasets = selected_datasets(args)
        runs = None

    # Teacher-assistant pre-stage, only if the tuned objective distils from an assistant.
    choice = tuning_choice(Paths(args.smoke)) if kcfg.get("use_tuning") else None
    if choice and choice["objective"].get("teacher") == "assistant":
        pre = [{**common, "dataset": ds, "task": t, "variant": "assistant", "run_name": "assistant",
                "seed": kcfg["teacher_seed"], "variant_spec": kcfg["assistant"], "make_cache": True}
               for ds in datasets for t in reg.tasks(ds, args.task)]
        run_all_specs(pre, args.jobs or kcfg["jobs"])

    specs = []
    for ds in datasets:
        for task in reg.tasks(ds, args.task):
            for s in seeds_of(ds):
                if runs is None:
                    for v in args.variants or reg.variants(ds, list(kcfg["variants"])):
                        specs.append({**common, "dataset": ds, "task": task, "variant": v, "seed": s})
                for v, frac, k in runs or []:
                    spec = {**common, "dataset": ds, "task": task, "variant": v, "seed": s, "k": k}
                    if frac is not None:
                        spec.update(run_name=f"{v}_f{frac}",
                                    variant_spec={**kcfg["variants"][v], "train_fraction": float(frac)})
                    specs.append(spec)
    run_all_specs(specs, args.jobs or kcfg["jobs"])


if __name__ == "__main__":
    main()
