"""Interim Parquet -> processed arrays (data/processed/<dataset>/) + outputs/data_report_<dataset>.json."""

from shield.cli import parser, paths_for, selected_datasets
from shield.data.preprocess import preprocess_botiot, preprocess_ciciot
from shield.utils.config import load_config
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("preprocess")


def main():
    args = parser(__doc__).parse_args()
    paths, cfg = paths_for(args), load_config("data", args.smoke)
    for ds in selected_datasets(args):
        out = paths.processed(ds)
        if (out / "meta.json").exists() and not args.force:
            log.info("%s already processed, skipping (use --force)", ds)
            continue
        fn = preprocess_ciciot if ds == "ciciot" else preprocess_botiot
        meta = fn(paths.interim(ds), out, cfg)
        report = {k: v for k, v in meta.items() if k != "scaler"}
        save_json(report, paths.out() / f"data_report_{ds}.json")


if __name__ == "__main__":
    main()
