"""Train the teacher for every (dataset, task, seed) in configs/datasets.yaml."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.models.train import train_teacher
from shield.utils.config import load_config
from shield.utils.device import get_device


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    paths, cfg, reg = paths_for(args), load_config("teacher", args.smoke), registry(args)
    device = get_device(args.cpu, args.smoke)
    for ds in selected_datasets(args):
        for task in reg.tasks(ds, args.task):
            for seed in args.seeds if args.seeds is not None else reg.seeds(ds, cfg["seeds"]):
                train_teacher(paths, ds, task, seed, cfg, device, args.force)


if __name__ == "__main__":
    main()
