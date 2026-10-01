"""Stream the raw CSVs straight out of the zips into Parquet (nothing is extracted to disk)."""

from __future__ import annotations

import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shield.utils.logging import get_logger

log = get_logger(__name__)

CICIOT_SPLITS = {
    "train": "CICIOT23/train/train.csv",
    "validation": "CICIOT23/validation/validation.csv",
    "test": "CICIOT23/test/test.csv",
}
CICIOT_LABEL = "label"

# Identity / time columns that leak the capture setup rather than describe traffic.
BOTIOT_DROP = ["pkSeqID", "stime", "ltime", "saddr", "daddr", "smac", "dmac",
               "soui", "doui", "sco", "dco", "sport", "dport"]
BOTIOT_CAT = ["flgs", "proto", "state"]
BOTIOT_NUM = ["pkts", "bytes", "seq", "dur", "mean", "stddev", "sum", "min", "max",
              "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]
BOTIOT_TARGETS = ["attack", "category", "subcategory"]

CICIOMT_SPLITS = ("train", "test")
LABEL = "label"  # attack label column written for the CIC datasets


def _write_chunks(chunks, out_path: Path) -> int:
    """Write an iterator of DataFrames to one Parquet file; returns the row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    writer, rows = None, 0
    try:
        for df in chunks:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(df)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out_path)
    return rows


def ingest_ciciot(zip_path: Path, out_dir: Path, chunksize: int, max_rows: int | None = None,
                  force: bool = False) -> dict[str, int]:
    counts = {}
    with zipfile.ZipFile(zip_path) as zf:
        for split, member in CICIOT_SPLITS.items():
            out = out_dir / f"{split}.parquet"
            if out.exists() and not force:
                counts[split] = pq.ParquetFile(out).metadata.num_rows
                log.info("CICIoT2023 %s: exists (%d rows), skipping", split, counts[split])
                continue

            def chunks():
                with zf.open(member) as f:
                    reader = pd.read_csv(f, chunksize=chunksize, nrows=max_rows, low_memory=False)
                    for df in reader:
                        df.columns = [c.strip() for c in df.columns]
                        feats = [c for c in df.columns if c != CICIOT_LABEL]
                        df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32)
                        df[CICIOT_LABEL] = df[CICIOT_LABEL].astype(str).str.strip()
                        yield df

            counts[split] = _write_chunks(chunks(), out)
            log.info("CICIoT2023 %s: %d rows -> %s", split, counts[split], out)
    return counts


def ciciomt_label(member: str) -> str:
    """'CICIoMT2024/train/TCP_IP-DDoS-ICMP3_train.pcap.csv' -> 'DDoS-ICMP' (numbered parts merged)."""
    m = re.fullmatch(r"(.+)_(train|test)\.pcap\.csv", Path(member.replace("\\", "/")).name)
    if not m:
        raise ValueError(f"Unexpected CICIoMT2024 file name: {member}")
    return re.sub(r"\d+$", "", m.group(1).removeprefix("TCP_IP-"))


def ciciomt_members(zip_path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".pcap.csv")]
    out = {s: sorted(n for n in names if f"/{s}/" in "/" + n.replace("\\", "/")) for s in CICIOMT_SPLITS}
    for s, members in out.items():
        if not members:
            raise FileNotFoundError(f"No CICIoMT2024 {s} CSVs in {zip_path}")
    return out


def ingest_ciciomt(zip_path: Path, out_dir: Path, chunksize: int, max_rows: int | None = None,
                   force: bool = False) -> dict[str, int]:
    """Wi-Fi/MQTT attack CSVs (capture-level train/test) -> one Parquet per split with a `label` column."""
    counts = {}
    members = ciciomt_members(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for split, files in members.items():
            out = out_dir / f"{split}.parquet"
            if out.exists() and not force:
                counts[split] = pq.ParquetFile(out).metadata.num_rows
                log.info("CICIoMT2024 %s: exists (%d rows), skipping", split, counts[split])
                continue

            def chunks():
                for member in files:
                    label = ciciomt_label(member)
                    with zf.open(member) as f:
                        for df in pd.read_csv(f, chunksize=chunksize, nrows=max_rows, low_memory=False):
                            df.columns = [c.strip() for c in df.columns]
                            df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
                            df[LABEL] = label
                            yield df

            counts[split] = _write_chunks(chunks(), out)
            log.info("CICIoMT2024 %s: %d rows from %d files -> %s", split, counts[split], len(files), out)
    return counts


def _botiot_member_index(name: str) -> int:
    m = re.fullmatch(r"data_(\d+)\.csv", Path(name).name)
    return int(m.group(1)) if m else -1


def botiot_members(zip_path: Path, max_files: int | None = None) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted((n for n in zf.namelist() if _botiot_member_index(n) > 0), key=_botiot_member_index)
    if max_files is not None and max_files < len(members):
        # Evenly spaced files so a small sample still spans the capture timeline (and its classes).
        idx = np.linspace(0, len(members) - 1, max_files).round().astype(int)
        members = [members[i] for i in sorted(set(idx))]
    return members


def _ingest_botiot_file(zip_path: str, member: str, out_path: str, chunksize: int,
                        max_rows: int | None) -> tuple[str, int]:
    keep = set(BOTIOT_NUM + BOTIOT_CAT + BOTIOT_TARGETS)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        header = f.readline().decode("utf-8").rstrip("\r\n").split(",")
    # Raw header names may carry stray whitespace (e.g. "subcategory "), so map raw -> stripped.
    raw = {h: h.strip() for h in header if h.strip() in keep}
    dtypes = {h: ("float32" if s in BOTIOT_NUM else str) for h, s in raw.items()}

    def chunks():
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
            reader = pd.read_csv(f, chunksize=chunksize, nrows=max_rows, low_memory=False,
                                 usecols=list(raw), dtype=dtypes)
            for df in reader:
                df = df.rename(columns=raw)
                for c in BOTIOT_CAT + ["category", "subcategory"]:
                    df[c] = df[c].fillna("").astype(str).str.strip()
                df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(np.int8)
                yield df[BOTIOT_NUM + BOTIOT_CAT + BOTIOT_TARGETS]

    rows = _write_chunks(chunks(), Path(out_path))
    return member, rows


def ingest_botiot(zip_path: Path, out_dir: Path, chunksize: int, workers: int,
                  max_rows_per_file: int | None = None, max_files: int | None = None,
                  force: bool = False) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    members = botiot_members(zip_path, max_files)
    jobs, counts = {}, {}
    for m in members:
        out = out_dir / f"part_{_botiot_member_index(m):03d}.parquet"
        if out.exists() and not force:
            counts[m] = pq.ParquetFile(out).metadata.num_rows
        else:
            jobs[m] = out
    log.info("Bot-IoT: %d files (%d already done), %d workers", len(members), len(counts), workers)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_ingest_botiot_file, str(zip_path), m, str(o), chunksize, max_rows_per_file)
                    for m, o in jobs.items()]
            for i, fut in enumerate(as_completed(futs), 1):
                member, rows = fut.result()
                counts[member] = rows
                log.info("Bot-IoT %s: %d rows (%d/%d)", member, rows, i, len(jobs))
    log.info("Bot-IoT total rows: %d", sum(counts.values()))
    return counts
