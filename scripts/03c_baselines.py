"""Tiny decision tree + logistic regression baselines for every (dataset, task) in the `baselines` stage."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.models.baselines import train_baseline
from shield.utils.config import load_config


def main():
    args = parser(__doc__).parse_args()
    paths, reg = paths_for(args), registry(args)
    cfg = load_config("teacher", args.smoke)["baselines"]
    for ds in selected_datasets(args, stage="baselines"):
        for task in reg.tasks(ds, args.task):
            for kind in cfg["models"]:
                train_baseline(paths, ds, task, kind, cfg, force=args.force)


if __name__ == "__main__":
    main()
