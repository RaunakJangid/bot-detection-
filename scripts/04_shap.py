"""SHAP analysis of every trained teacher, plus ranking stability across teacher seeds."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.utils.config import load_config
from shield.utils.device import get_device
from shield.xai.shap_analysis import run_shap, shap_stability


def main():
    args = parser(__doc__).parse_args()
    paths, reg = paths_for(args), registry(args)
    tcfg, scfg = load_config("teacher", args.smoke), load_config("shap", args.smoke)
    device = get_device(args.cpu, args.smoke)
    for ds in selected_datasets(args):
        seeds = reg.seeds(ds, tcfg["seeds"])
        for task in reg.tasks(ds, args.task):
            for seed in seeds:
                run_shap(paths, ds, task, tcfg["model"], seed, scfg, device, args.force)
            shap_stability(paths, ds, task, tcfg["model"], seeds)


if __name__ == "__main__":
    main()
