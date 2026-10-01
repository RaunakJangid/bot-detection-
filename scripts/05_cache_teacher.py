"""Cache the KD teacher's logits and gradient x input over the training split."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.kd.cache import build_cache
from shield.utils.config import load_config
from shield.utils.device import get_device


def main():
    args = parser(__doc__).parse_args()
    paths, reg = paths_for(args), registry(args)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    device = get_device(args.cpu, args.smoke)
    for ds in selected_datasets(args):
        for task in reg.tasks(ds, args.task):
            build_cache(paths, ds, task, tcfg["model"], kcfg["teacher_seed"], device, force=args.force)


if __name__ == "__main__":
    main()
