"""Train the XGBoost reference model for every (dataset, task)."""

from shield.cli import parser, paths_for, selected_datasets, tasks_for
from shield.models.xgb import gpu_available, train_xgboost
from shield.utils.config import load_config


def main():
    args = parser(__doc__).parse_args()
    paths, cfg = paths_for(args), load_config("teacher", args.smoke)
    use_gpu = gpu_available() and not args.cpu
    for ds in selected_datasets(args):
        for task in tasks_for(cfg["tasks"], ds, args.task):
            for seed in cfg["xgboost"]["seeds"]:
                train_xgboost(paths, ds, task, seed, cfg["xgboost"], use_gpu, args.force)


if __name__ == "__main__":
    main()
