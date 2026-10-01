"""Interim Parquet -> processed arrays (data/processed/<dataset>/) + outputs/data_report_<dataset>.json."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.data.preprocess import preprocess_dataset
from shield.data.registry import SOURCES
from shield.utils.config import load_config
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("preprocess")


def main():
    args = parser(__doc__).parse_args()
    paths, cfg, reg = paths_for(args), load_config("data", args.smoke), registry(args)
    interims = {s: paths.interim(s) for s in SOURCES}
    for ds in selected_datasets(args):
        out = paths.processed(ds)
        if (out / "meta.json").exists() and not args.force:
            log.info("%s already processed, skipping (use --force)", ds)
            continue
        drop = reg.dropped(ds)
        log.info("Preprocessing %s (source %s, dropping %s)", ds, reg.source(ds), drop or "nothing")
        meta = preprocess_dataset(ds, reg.spec(ds), drop, interims, out, cfg)
        save_json({k: v for k, v in meta.items() if k != "scaler"}, paths.out() / f"data_report_{ds}.json")


if __name__ == "__main__":
    main()
