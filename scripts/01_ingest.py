"""Stream the dataset zips into Parquet (data/interim/<dataset>/)."""

from shield.cli import parser, paths_for, selected_datasets
from shield.data.ingest import ingest_botiot, ingest_ciciot
from shield.utils.config import load_config


def main():
    args = parser(__doc__).parse_args()
    paths, cfg = paths_for(args), load_config("data", args.smoke)
    for ds in selected_datasets(args):
        if ds == "ciciot":
            ingest_ciciot(paths.ciciot_zip, paths.interim("ciciot"), cfg["chunksize"], cfg["max_rows_per_file"],
                          args.force)
        else:
            ingest_botiot(paths.botiot_zip, paths.interim("botiot"), cfg["chunksize"], cfg["botiot_workers"],
                          cfg["max_rows_per_file"], cfg["max_botiot_files"], args.force)


if __name__ == "__main__":
    main()
