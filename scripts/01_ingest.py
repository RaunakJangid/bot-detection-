"""Stream the dataset zips into Parquet (data/interim/<source>/) for every source the selected datasets need."""

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.data.ingest import ingest_botiot, ingest_ciciomt, ingest_ciciot, ingest_ciciot_full
from shield.utils.config import load_config


def main():
    args = parser(__doc__).parse_args()
    paths, cfg, reg = paths_for(args), load_config("data", args.smoke), registry(args)
    sources = []
    for ds in selected_datasets(args):
        spec = reg.spec(ds)
        # shared (cross) datasets need both CIC sources
        for src in (["ciciot", "ciciomt"] if spec.get("shared") else [spec["source"]]):
            if src not in sources:
                sources.append(src)
    for src in sources:
        if src == "ciciot":
            ingest_ciciot(paths.ciciot_zip, paths.interim("ciciot"), cfg["chunksize"], cfg["max_rows_per_file"],
                          args.force)
        elif src == "ciciot_full":
            if paths.ciciot_full is None:
                raise SystemExit("set ciciot_full in configs/paths.yaml (zip or folder of the 169 part files)")
            ingest_ciciot_full(paths.ciciot_full, paths.interim("ciciot_full"), cfg["chunksize"], args.force)
        elif src == "ciciomt":
            ingest_ciciomt(paths.ciciomt_zip, paths.interim("ciciomt"), cfg["chunksize"],
                           cfg["ciciomt_max_rows_per_file"], args.force)
        else:
            ingest_botiot(paths.botiot_zip, paths.interim("botiot"), cfg["chunksize"], cfg["botiot_workers"],
                          cfg["max_rows_per_file"], cfg["max_botiot_files"], args.force)


if __name__ == "__main__":
    main()
