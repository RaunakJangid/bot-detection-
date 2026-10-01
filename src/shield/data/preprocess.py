"""Interim Parquet -> model-ready arrays.

Output layout (data/processed/<dataset>/):
    X_<split>.npy              float32 (rows, features), imputed + transformed + standardised
    y_<task>_<split>.npy       int16 class indices
    meta.json                  features, classes per task, split counts, scaler statistics
Splits are always: train, validation, test. All statistics are fitted on train only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from shield.data.ingest import BOTIOT_CAT, BOTIOT_NUM, CICIOT_LABEL, HASH_COL
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)

SPLITS = ("train", "validation", "test")
BOTIOT_CATEGORIES = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]
CICIOT_FAMILIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
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
    return np.sort(np.concatenate(keep))


def stratified_split(strata: np.ndarray, fracs: tuple[float, float, float],
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-stratum permutation split. Returns sorted positional indices per split."""
    parts = {s: [] for s in SPLITS}
    for c in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == c))
        n = len(members)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        parts["train"].append(members[:n_tr])
        parts["validation"].append(members[n_tr:n_tr + n_va])
        parts["test"].append(members[n_tr + n_va:])
    return {s: np.sort(np.concatenate(v)) for s, v in parts.items()}


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


# ---------------------------------------------------------------- CICIoT2023

def preprocess_ciciot(interim: Path, out_dir: Path, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    frames = {s: pd.read_parquet(interim / f"{s}.parquet") for s in SPLITS}
    features = [c for c in frames["train"].columns if c != CICIOT_LABEL]
    # Union over splits: a capped or partial train split can miss rare classes that val/test contain.
    classes34 = sorted(set().union(*(set(df[CICIOT_LABEL].unique()) for df in frames.values())))
    families_present = {ciciot_family(c) for c in classes34}
    families = [f for f in CICIOT_FAMILIES if f in families_present]
    tasks = {"binary": ["Benign", "Attack"], "family": families, "class34": classes34}
    log.info("CICIoT2023: %d features, %d classes, %d families", len(features), len(classes34), len(families))

    cls_index = {c: i for i, c in enumerate(classes34)}
    fam_index = {f: i for i, f in enumerate(families)}
    raw, ys = {}, {t: {} for t in tasks}
    missing = set(classes34) - set(frames["train"][CICIOT_LABEL].unique())
    if missing:
        log.warning("CICIoT2023 train split lacks classes %s (expected only for partial/smoke data)", sorted(missing))
    for split, df in frames.items():
        y34 = df[CICIOT_LABEL].map(cls_index).to_numpy(np.int16)
        keep = cap_per_class(np.arange(len(df)), y34, cfg.get("max_per_class"), rng)
        fam = np.array([fam_index[ciciot_family(l)] for l in classes34], dtype=np.int16)[y34[keep]]
        ys["class34"][split] = y34[keep]
        ys["family"][split] = fam
        ys["binary"][split] = (fam != fam_index["Benign"]).astype(np.int16)
        raw[split] = df[features].to_numpy(np.float32)[keep]
    del frames

    scaler = NumericScaler(cfg["transform"]).fit(raw["train"])
    X = {}
    for split in SPLITS:
        X[split] = np.empty_like(raw[split])
        scaler.apply(raw[split], X[split])
        del raw[split]
    meta = {"dataset": "ciciot", "features": features, "n_features": len(features),
            "n_rows": {s: len(X[s]) for s in SPLITS}, "scaler": scaler.state(),
            "max_per_class": cfg.get("max_per_class")}
    _write_outputs(out_dir, X, ys, tasks, meta)
    return meta


# ---------------------------------------------------------------- Bot-IoT

def _botiot_parts(interim: Path) -> list[Path]:
    parts = sorted(interim.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Bot-IoT parts in {interim}; run 01_ingest.py first")
    return parts


def dedup_mask(h: np.ndarray, labkey: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep the first row of every distinct feature hash; drop every row of hashes whose labels conflict."""
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


def preprocess_botiot(interim: Path, out_dir: Path, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    parts = _botiot_parts(interim)
    cat_index = {c: i for i, c in enumerate(BOTIOT_CATEGORIES)}

    # Pass 1: hashes + labels only (small), for de-duplication and the stratified split.
    hs, cats, subs, attack, sizes = [], [], [], [], []
    sub_vocab: dict[str, int] = {}
    for p in parts:
        t = pq.read_table(p, columns=[HASH_COL, "category", "subcategory", "attack"]).to_pandas()
        unknown = set(t["category"].unique()) - set(cat_index)
        if unknown:
            raise ValueError(f"Unknown Bot-IoT categories {sorted(unknown)} in {p.name}")
        hs.append(t[HASH_COL].to_numpy(np.uint64))
        cats.append(t["category"].map(cat_index).to_numpy(np.int16))
        for s in t["subcategory"].unique():
            sub_vocab.setdefault(s, len(sub_vocab))
        subs.append(t["subcategory"].map(sub_vocab).to_numpy(np.int16))
        attack.append(t["attack"].to_numpy(np.int16))
        sizes.append(len(t))
    h = np.concatenate(hs); cat = np.concatenate(cats); sub = np.concatenate(subs); att = np.concatenate(attack)
    del hs, cats, subs, attack
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    log.info("Bot-IoT: %d rows across %d parts", len(h), len(parts))

    labkey = cat.astype(np.int64) * 1024 + sub
    keep, dedup_stats = dedup_mask(h, labkey)
    del h, labkey
    log.info("Bot-IoT de-dup: %s", dedup_stats)

    kept = np.flatnonzero(keep)
    split_pos = stratified_split(sub[kept], tuple(cfg["botiot_split"]), rng)
    split_idx = {}
    for s in SPLITS:
        gidx = kept[split_pos[s]]
        split_idx[s] = cap_per_class(gidx, cat[gidx], cfg.get("max_per_class"), rng)
    del kept, keep

    # Pass 2: gather numeric features + categorical codes for the selected rows, part by part.
    cat_vocab = {c: {} for c in BOTIOT_CAT}
    num = {s: np.empty((len(split_idx[s]), len(BOTIOT_NUM)), np.float32) for s in SPLITS}
    codes = {s: np.empty((len(split_idx[s]), len(BOTIOT_CAT)), np.int16) for s in SPLITS}
    for pi, p in enumerate(parts):
        lo, hi = offsets[pi], offsets[pi + 1]
        df = pq.read_table(p, columns=BOTIOT_NUM + BOTIOT_CAT).to_pandas()
        for j, c in enumerate(BOTIOT_CAT):
            for v in df[c].unique():
                cat_vocab[c].setdefault(v, len(cat_vocab[c]))
        part_codes = np.stack([df[c].map(cat_vocab[c]).to_numpy(np.int16) for c in BOTIOT_CAT], axis=1)
        part_num = df[BOTIOT_NUM].to_numpy(np.float32)
        del df
        for s in SPLITS:
            a, b = np.searchsorted(split_idx[s], [lo, hi])
            local = split_idx[s][a:b] - lo
            num[s][a:b] = part_num[local]
            codes[s][a:b] = part_codes[local]
        log.info("Bot-IoT gather %d/%d", pi + 1, len(parts))

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
    features = BOTIOT_NUM + onehot_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        X = np.lib.format.open_memmap(out_dir / f"X_{s}.npy", mode="w+", dtype=np.float32,
                                      shape=(len(split_idx[s]), len(features)))
        scaler.apply(num[s], X[:, :len(BOTIOT_NUM)])
        X[:, len(BOTIOT_NUM):] = 0.0
        for j, (base, col_of) in enumerate(onehot_maps):
            cols = len(BOTIOT_NUM) + base + col_of[codes[s][:, j]]
            X[np.arange(len(X)), cols] = 1.0
        X.flush(); del X
        num[s] = None

    tasks = {"binary": ["Normal", "Attack"], "category": BOTIOT_CATEGORIES}
    ys = {"binary": {}, "category": {}}
    for s in SPLITS:
        ys["binary"][s] = att[split_idx[s]]
        ys["category"][s] = cat[split_idx[s]]
    meta = {"dataset": "botiot", "features": features, "n_features": len(features),
            "numeric_features": BOTIOT_NUM, "onehot_features": onehot_cols,
            "n_rows": {s: int(len(split_idx[s])) for s in SPLITS}, "scaler": scaler.state(),
            "dedup": dedup_stats, "subcategories": list(sub_vocab), "max_per_class": cfg.get("max_per_class")}
    _write_outputs(out_dir, None, ys, tasks, meta)
    return meta
