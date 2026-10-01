"""Train the teacher for every (dataset, task, seed) in configs/teacher.yaml."""

from shield.cli import parser, paths_for, selected_datasets, tasks_for
from shield.models.train import train_teacher
from shield.utils.config import load_config
from shield.utils.device import get_device


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    paths, cfg = paths_for(args), load_config("teacher", args.smoke)
    device = get_device(args.cpu, args.smoke)
    for ds in selected_datasets(args):
        for task in tasks_for(cfg["tasks"], ds, args.task):
            for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
                train_teacher(paths, ds, task, seed, cfg, device, args.force)


if __name__ == "__main__":
    main()
