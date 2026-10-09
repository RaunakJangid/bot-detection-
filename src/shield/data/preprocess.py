"""Interim Parquet -> model-ready arrays.

Output layout (data/processed/<dataset>/):
    X_<split>.npy              float32 (rows, features), imputed + transformed + standardised
    y_<task>_<split>.npy       int16 class indices
    meta.json                  features, classes per task, split counts, scaler statistics, de-dup stats
Splits are always: train, validation, test. All statistics are fitted on train only.

Leakage controls (all datasets):
  * `drop`: features removed under the strict protocol (configs/datasets.yaml).
  * Exact duplicate feature vectors are removed across splits (train kept first), so no test row
    also appears in train; hashes with conflicting labels are dropped entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from shield.data.ingest import BOTIOT_CAT, BOTIOT_NUM, LABEL
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)

SPLITS = ("train", "validation", "test")
BOTIOT_CATEGORIES = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]
CICIOT_FAMILIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
CICIOMT_CATEGORIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SHARED5 = ["Benign", "DDoS", "DoS", "Recon", "Spoofing"]
_CICIOT_EXACT = {
    "VulnerabilityScan": "Recon",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    "SqlInjection": "Web",
    "XSS": "Web",
    "CommandInjection": "Web",
    "Backdoor_Malware": "Web",
    "Uploading_Attack": "Web",
    "BrowserHijacking": "Web",
    "DictionaryBruteForce": "BruteForce",
}
_CICIOT_PREFIX = [("Benign", "Benign"), ("DDoS", "DDoS"), ("DoS", "DoS"), ("Mirai", "Mirai"), ("Recon", "Recon")]


def ciciot_family(label: str) -> str:
    if label in _CICIOT_EXACT:
        return _CICIOT_EXACT[label]
    for prefix, family in _CICIOT_PREFIX:
        if label.startswith(prefix):
            return family
    raise ValueError(f"Unknown CICIoT2023 label {label!r}: add it to the family mapping in preprocess.py")


def ciciomt_category(label: str) -> str:
    if label == "Benign":
        return "Benign"
    for prefix in ("MQTT", "DDoS", "DoS", "Recon"):  # MQTT first: "MQTT-DDoS-..." is an MQTT attack
        if label.startswith(prefix):
            return prefix
    if "Spoofing" in label:
        return "Spoofing"
    raise ValueError(f"Unknown CICIoMT2024 label {label!r}: add it to ciciomt_category in preprocess.py")


def family_of(source: str, label: str) -> str:
    return ciciot_family(label) if source.startswith("ciciot") else ciciomt_category(label)


def shared_class(source: str, label: str) -> str | None:
    """Class in the label space both CIC datasets share, or None (Mirai, Web, BruteForce, MQTT)."""
    fam = family_of(source, label)
    return fam if fam in SHARED5 else None


TaskSpec = tuple[list[str], Callable[[str], str]]


def cic_task_specs(source: str, labels: list[str], shared: bool) -> dict[str, TaskSpec]:
    benign = lambda l: "Benign" if family_of(source, l) == "Benign" else "Attack"
    if shared:
        return {"binary": (["Benign", "Attack"], benign),
                "shared5": (SHARED5, lambda l: shared_class(source, l))}
    present = {family_of(source, l) for l in labels}
    if source.startswith("ciciot"):
        fams = [f for f in CICIOT_FAMILIES if f in present]
        return {"binary": (["Benign", "Attack"], benign), "family": (fams, ciciot_family),
                "class34": (sorted(labels), lambda l: l)}
    cats = [c for c in CICIOMT_CATEGORIES if c in present]
    return {"binary": (["Benign", "Attack"], benign), "category": (cats, ciciomt_category),
            "attack": (sorted(labels), lambda l: l)}


# ---------------------------------------------------------------- numeric transform

def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class NumericScaler:
    """Median imputation (inf -> NaN) + optional signed-log + standardisation, fitted on train."""

    def __init__(self, transform: str = "signed_log"):
        self.transform = transform

    def _pre(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        x[~np.isfinite(x)] = np.nan
        x = np.where(np.isnan(x), self.median, x)
        return signed_log(x) if self.transform == "signed_log" else x

    def fit(self, train: np.ndarray, chunk: int = 2_000_000) -> "NumericScaler":
        clean = train.astype(np.float32, copy=True)
        clean[~np.isfinite(clean)] = np.nan
        self.median = np.nan_to_num(np.nanmedian(clean, axis=0), nan=0.0).astype(np.float32)
        del clean
        s = np.zeros(train.shape[1]); ss = np.zeros(train.shape[1]); n = 0
        for i in range(0, len(train), chunk):
            z = self._pre(train[i:i + chunk]).astype(np.float64)
            s += z.sum(0); ss += (z * z).sum(0); n += len(z)
        self.mean = (s / n).astype(np.float32)
        var = np.maximum(ss / n - (s / n) ** 2, 0.0)
        self.var = var.astype(np.float32)  # variance after the transform (used by variance selection)
        self.std = np.where(var > 1e-12, np.sqrt(var), 1.0).astype(np.float32)
        return self

    def apply(self, x: np.ndarray, out: np.ndarray, chunk: int = 2_000_000) -> None:
        for i in range(0, len(x), chunk):
            out[i:i + chunk] = (self._pre(x[i:i + chunk]) - self.mean) / self.std

    def state(self) -> dict:
        return {"transform": self.transform, "median": self.median, "mean": self.mean,
                "std": self.std, "var": self.var}


# ---------------------------------------------------------------- helpers

def cap_per_class(idx: np.ndarray, y: np.ndarray, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Keep at most `cap` rows per class (all rows of smaller classes). Returns sorted indices."""
    if cap is None:
        return idx
    keep = []
    for c in np.unique(y):
        members = idx[y == c]
        keep.append(members if len(members) <= cap else rng.choice(members, cap, replace=False))
    return np.sort(np.concatenate(keep)) if keep else idx


def stratified_split(strata: np.ndarray, fracs: tuple[float, float, float],
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-stratum permutation split. Returns sorted positional indices per split.
    With fracs[2] == 0 the validation split takes the remainder (nothing is lost to rounding)."""
    parts = {s: [] for s in SPLITS}
    for c in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == c))
        n = len(members)
        n_tr = int(round(fracs[0] * n))
        n_va = n - n_tr if fracs[2] == 0 else int(round(fracs[1] * n))
        parts["train"].append(members[:n_tr])
        parts["validation"].append(members[n_tr:n_tr + n_va])
        parts["test"].append(members[n_tr + n_va:])
    return {s: np.sort(np.concatenate(v)).astype(np.int64) for s, v in parts.items()}


def row_hash(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy(np.uint64)


def dedup_mask(h: np.ndarray, labkey: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep the first row (in input order) of every distinct feature hash; drop every row of hashes
    whose labels conflict."""
    order = np.argsort(h, kind="stable")
    hs, lab = h[order], labkey[order]
    start = np.empty(len(hs), dtype=bool)
    start[0] = True
    np.not_equal(hs[1:], hs[:-1], out=start[1:])
    gid = np.cumsum(start) - 1
    first_lab = lab[start][gid]
    conflict_group = np.zeros(gid[-1] + 1, dtype=bool)
    conflict_group[gid[lab != first_lab]] = True
    keep_sorted = start & ~conflict_group[gid]
    keep = np.zeros(len(h), dtype=bool)
    keep[order[keep_sorted]] = True
    stats = {
        "rows_in": int(len(h)),
        "unique_feature_rows": int(start.sum()),
        "duplicate_rows_dropped": int(len(h) - start.sum()),
        "conflicting_hashes": int(conflict_group.sum()),
        "conflicting_rows_dropped": int(conflict_group[gid].sum()),
        "rows_out": int(keep.sum()),
    }
    return keep, stats


def dedup_across_splits(hashes: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """De-duplicate over train -> validation -> test, so a vector seen in train is removed from
    validation/test (and from later positions within a split). Returns (keep mask per split, stats)."""
    h = np.concatenate([hashes[s] for s in SPLITS])
    lab = np.concatenate([labels[s] for s in SPLITS]).astype(np.int64)
    keep, stats = dedup_mask(h, lab)
    out, start = {}, 0
    for s in SPLITS:
        n = len(hashes[s])
        out[s] = keep[start:start + n]
        stats[f"{s}_in"], stats[f"{s}_out"] = int(n), int(out[s].sum())
        start += n
    return out, stats


def _class_counts(y: np.ndarray, classes: list[str]) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(classes))
    return {c: int(n) for c, n in zip(classes, counts)}


def _write_outputs(out_dir: Path, X: dict[str, np.ndarray] | None, ys: dict[str, dict[str, np.ndarray]],
                   tasks: dict[str, list[str]], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if X is not None:
        for split, arr in X.items():
            np.save(out_dir / f"X_{split}.npy", arr)
    report = {}
    for task, per_split in ys.items():
        report[task] = {}
        for split, y in per_split.items():
            np.save(out_dir / f"y_{task}_{split}.npy", y.astype(np.int16))
            report[task][split] = _class_counts(y, tasks[task])
    meta["tasks"] = tasks
    meta["class_counts"] = report
    save_json(meta, out_dir / "meta.json")
    log.info("Wrote %s", out_dir)


# ---------------------------------------------------------------- CICIoT2023 / CICIoMT2024

def load_cic_frames(source: str, interim: Path, cfg: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    if source == "ciciot":
        return {s: pd.read_parquet(interim / f"{s}.parquet") for s in SPLITS}
    if source == "ciciot_full":   # train: the full merged CSVs; validation/test: the strict subset's own files
        sub = interim.parent / "ciciot"
        full = pd.read_parquet(interim / "all.parquet")
        full[LABEL] = full[LABEL].astype("category")   # 46.6M label strings would cost ~3 GB
        return {"train": full,
                "validation": pd.read_parquet(sub / "validation.parquet"), "test": pd.read_parquet(sub / "test.parquet")}
    # CICIoMT2024: capture-level train/test as shipped; validation is a stratified slice of train.
    train, test = pd.read_parquet(interim / "train.parquet"), pd.read_parquet(interim / "test.parquet")
    v = cfg["ciciomt_val_frac"]
    parts = stratified_split(pd.factorize(train[LABEL])[0], (1 - v, v, 0.0), rng)
    return {"train": train.iloc[parts["train"]].reset_index(drop=True),
            "validation": train.iloc[parts["validation"]].reset_index(drop=True), "test": test}


def cic_feature_columns(interim: Path) -> list[str]:
    f = interim / "train.parquet"
    return [c for c in pq.ParquetFile(f if f.exists() else interim / "all.parquet").schema.names if c != LABEL]


def shared_features(interims: dict[str, Path], drop: list[str]) -> list[str]:
    """Features both CIC datasets have (CICIoT2023 column order), minus `drop`."""
    other = set(cic_feature_columns(interims["ciciomt"]))
    return [c for c in cic_feature_columns(interims["ciciot"]) if c in other and c not in drop]


def preprocess_cic(dataset: str, source: str, interim: Path, out_dir: Path, cfg: dict, drop: list[str],
                   shared: bool = False, features: list[str] | None = None) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    frames = load_cic_frames(source, interim, cfg, rng)
    all_cols = [c for c in frames["train"].columns if c != LABEL]
    features = features or [c for c in all_cols if c not in drop]
    missing = set(features) - set(all_cols)
    if missing:
        raise ValueError(f"{dataset}: features {sorted(missing)} not in {source}")

    labels = sorted(set().union(*(set(df[LABEL].unique()) for df in frames.values())))
    if shared:  # keep only classes present in both datasets' label spaces
        keep_labels = {l for l in labels if shared_class(source, l) is not None}
        frames = {s: df[df[LABEL].isin(keep_labels)].reset_index(drop=True) for s, df in frames.items()}
        labels = sorted(keep_labels)
    tasks = cic_task_specs(source, labels, shared)
    absent = set(labels) - set(frames["train"][LABEL].unique())
    if absent:
        log.warning("%s train split lacks classes %s (expected only for partial/smoke data)", dataset, sorted(absent))

    lab_index = {l: i for i, l in enumerate(labels)}
    codes = {s: df[LABEL].map(lab_index).to_numpy(np.int64) for s, df in frames.items()}
    hashes = {s: row_hash(df[features]) for s, df in frames.items()}
    keep, dedup_stats = dedup_across_splits(hashes, codes)
    if source == "ciciot_full":
        # Directly comparable with the strict CICIoT2023 result: validation/test are EXACTLY the rows the
        # strict subset kept (its train -> validation -> test de-dup is recomputed with the same code), and
        # the full-data train set drops every vector that occurs anywhere in the raw validation/test files.
        sub_train = pd.read_parquet(interim.parent / "ciciot" / "train.parquet")
        sub_h = {"train": row_hash(sub_train[features]), "validation": hashes["validation"], "test": hashes["test"]}
        sub_codes = {"train": sub_train[LABEL].map(lab_index).to_numpy(np.int64),
                     "validation": codes["validation"], "test": codes["test"]}
        del sub_train
        sub_keep, _ = dedup_across_splits(sub_h, sub_codes)
        keep["validation"], keep["test"] = sub_keep["validation"], sub_keep["test"]
        tr_keep, tr_stats = dedup_mask(hashes["train"], codes["train"])
        in_eval = np.isin(hashes["train"], np.concatenate([hashes["validation"], hashes["test"]]))
        keep["train"] = tr_keep & ~in_eval
        dedup_stats = {"train_internal": tr_stats, "train_rows_matching_eval_dropped": int((tr_keep & in_eval).sum()),
                       "train_out": int(keep["train"].sum()), "validation_out": int(keep["validation"].sum()),
                       "test_out": int(keep["test"].sum()), "eval_rows": "identical to the strict ciciot dataset"}
    log.info("%s de-dup across splits: %s", dataset, dedup_stats)

    raw, ys = {}, {t: {} for t in tasks}
    for s in SPLITS:
        idx = cap_per_class(np.flatnonzero(keep[s]), codes[s][keep[s]], cfg.get("max_per_class"), rng)
        df = frames[s].iloc[idx]
        for task, (classes, fn) in tasks.items():
            cls_index = {c: i for i, c in enumerate(classes)}
            lut = {l: cls_index[fn(l)] for l in labels}
            ys[task][s] = df[LABEL].map(lut).to_numpy(np.int16)
        raw[s] = df[features].to_numpy(np.float32)
    del frames

    scaler = NumericScaler(cfg["transform"]).fit(raw["train"])
    X = {}
    for s in SPLITS:
        X[s] = np.empty_like(raw[s])
        scaler.apply(raw[s], X[s])
        del raw[s]
    meta = {"dataset": dataset, "source": source, "shared": shared, "dropped_features": drop,
            "features": features, "n_features": len(features), "n_rows": {s: len(X[s]) for s in SPLITS},
            "labels": labels, "scaler": scaler.state(), "dedup": dedup_stats,
            "max_per_class": cfg.get("max_per_class")}
    _write_outputs(out_dir, X, ys, {t: c for t, (c, _) in tasks.items()}, meta)
    return meta


def preprocess_ciciot(interim: Path, out_dir: Path, cfg: dict, drop: list[str] = ()) -> dict:
    return preprocess_cic("ciciot", "ciciot", interim, out_dir, cfg, list(drop))


def preprocess_ciciomt(interim: Path, out_dir: Path, cfg: dict, drop: list[str] = ()) -> dict:
    return preprocess_cic("ciciomt", "ciciomt", interim, out_dir, cfg, list(drop))


# ---------------------------------------------------------------- Bot-IoT

def _botiot_parts(interim: Path) -> list[Path]:
    parts = sorted(interim.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Bot-IoT parts in {interim}; run 01_ingest.py first")
    return parts


def preprocess_botiot(interim: Path, out_dir: Path, cfg: dict, drop: list[str] = (),
                      dataset: str = "botiot") -> dict:
    rng = np.random.default_rng(cfg["seed"])
    parts = _botiot_parts(interim)
    cat_index = {c: i for i, c in enumerate(BOTIOT_CATEGORIES)}
    num_cols = [c for c in BOTIOT_NUM if c not in drop]

    # Pass 1: feature hashes (on the kept features) + labels, for de-duplication and the split.
    hs, cats, subs, attack, sizes = [], [], [], [], []
    sub_vocab: dict[str, int] = {}
    for p in parts:
        t = pq.read_table(p, columns=num_cols + BOTIOT_CAT + ["category", "subcategory", "attack"]).to_pandas()
        unknown = set(t["category"].unique()) - set(cat_index)
        if unknown:
            raise ValueError(f"Unknown Bot-IoT categories {sorted(unknown)} in {p.name}")
        hs.append(row_hash(t[num_cols + BOTIOT_CAT]))
        cats.append(t["category"].map(cat_index).to_numpy(np.int16))
        for s in t["subcategory"].unique():
            sub_vocab.setdefault(s, len(sub_vocab))
        subs.append(t["subcategory"].map(sub_vocab).to_numpy(np.int16))
        attack.append(t["attack"].to_numpy(np.int16))
        sizes.append(len(t))
    h = np.concatenate(hs); cat = np.concatenate(cats); sub = np.concatenate(subs); att = np.concatenate(attack)
    del hs, cats, subs, attack
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    log.info("%s: %d rows across %d parts", dataset, len(h), len(parts))

    labkey = cat.astype(np.int64) * 1024 + sub
    keep, dedup_stats = dedup_mask(h, labkey)
    del h, labkey
    log.info("%s de-dup: %s", dataset, dedup_stats)

    kept = np.flatnonzero(keep)
    split_pos = stratified_split(sub[kept], tuple(cfg["botiot_split"]), rng)
    split_idx = {}
    for s in SPLITS:
        gidx = kept[split_pos[s]]
        split_idx[s] = cap_per_class(gidx, cat[gidx], cfg.get("max_per_class"), rng)
    del kept, keep

    # Pass 2: gather numeric features + categorical codes for the selected rows, part by part.
    cat_vocab = {c: {} for c in BOTIOT_CAT}
    num = {s: np.empty((len(split_idx[s]), len(num_cols)), np.float32) for s in SPLITS}
    codes = {s: np.empty((len(split_idx[s]), len(BOTIOT_CAT)), np.int16) for s in SPLITS}
    for pi, p in enumerate(parts):
        lo, hi = offsets[pi], offsets[pi + 1]
        df = pq.read_table(p, columns=num_cols + BOTIOT_CAT).to_pandas()
        for c in BOTIOT_CAT:
            for v in df[c].unique():
                cat_vocab[c].setdefault(v, len(cat_vocab[c]))
        part_codes = np.stack([df[c].map(cat_vocab[c]).to_numpy(np.int16) for c in BOTIOT_CAT], axis=1)
        part_num = df[num_cols].to_numpy(np.float32)
        del df
        for s in SPLITS:
            a, b = np.searchsorted(split_idx[s], [lo, hi])
            local = split_idx[s][a:b] - lo
            num[s][a:b] = part_num[local]
            codes[s][a:b] = part_codes[local]
        log.info("%s gather %d/%d", dataset, pi + 1, len(parts))

    # One-hot vocabulary from train counts; rare values fold into "<col>=other".
    onehot_cols, onehot_maps = [], []
    for j, c in enumerate(BOTIOT_CAT):
        inv = {i: v for v, i in cat_vocab[c].items()}
        counts = np.bincount(codes["train"][:, j], minlength=len(inv))
        frequent = [i for i in range(len(inv)) if counts[i] >= cfg["min_category_count"]]
        col_of = np.full(len(inv), -1, np.int32)
        names = []
        for i in frequent:
            col_of[i] = len(names); names.append(f"{c}={inv[i] or '<empty>'}")
        other = len(names); names.append(f"{c}=other")
        col_of[col_of < 0] = other
        onehot_cols.extend(names); onehot_maps.append((len(onehot_cols) - len(names), col_of))

    scaler = NumericScaler(cfg["transform"]).fit(num["train"])
    features = num_cols + onehot_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        X = np.lib.format.open_memmap(out_dir / f"X_{s}.npy", mode="w+", dtype=np.float32,
                                      shape=(len(split_idx[s]), len(features)))
        scaler.apply(num[s], X[:, :len(num_cols)])
        X[:, len(num_cols):] = 0.0
        for j, (base, col_of) in enumerate(onehot_maps):
            cols = len(num_cols) + base + col_of[codes[s][:, j]]
            X[np.arange(len(X)), cols] = 1.0
        X.flush(); del X
        num[s] = None

    tasks = {"binary": ["Normal", "Attack"], "category": BOTIOT_CATEGORIES}
    ys = {"binary": {}, "category": {}}
    for s in SPLITS:
        ys["binary"][s] = att[split_idx[s]]
        ys["category"][s] = cat[split_idx[s]]
    meta = {"dataset": dataset, "source": "botiot", "shared": False, "dropped_features": list(drop),
            "features": features, "n_features": len(features),
            "numeric_features": num_cols, "onehot_features": onehot_cols,
            "n_rows": {s: int(len(split_idx[s])) for s in SPLITS}, "scaler": scaler.state(),
            "dedup": dedup_stats, "subcategories": list(sub_vocab), "max_per_class": cfg.get("max_per_class")}
    _write_outputs(out_dir, None, ys, tasks, meta)
    return meta


# ---------------------------------------------------------------- dispatch

def preprocess_dataset(dataset: str, spec: dict, drop: list[str], interims: dict[str, Path], out_dir: Path,
                       cfg: dict) -> dict:
    source = spec["source"]
    if source == "botiot":
        return preprocess_botiot(interims["botiot"], out_dir, cfg, drop, dataset)
    features = shared_features(interims, drop) if spec.get("shared") else None
    return preprocess_cic(dataset, source, interims[source], out_dir, cfg, drop, bool(spec.get("shared")), features)
