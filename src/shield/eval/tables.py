"""Paper tables (CSV + LaTeX) built from the saved run outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shield.utils.config import Paths
from shield.utils.io import load_json


def _pm(values: list[float], scale: float = 100.0, digits: int = 2) -> str:
    v = np.asarray(values, dtype=float) * scale
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def _metric_rows(files: list[Path], key: str = "test") -> dict[str, list[float]]:
    out = {"macro_f1": [], "accuracy": [], "mcc": []}
    for f in files:
        m = load_json(f)[key]
        for k in out:
            out[k].append(m.get(k, np.nan))
    return out


def detection_table(paths: Paths, dataset: str, task: str, teacher_model: str, variants: list[str]) -> pd.DataFrame:
    lat = {}
    lat_file = paths.outputs / "latency" / "latency.json"
    if lat_file.exists():
        lat = load_json(lat_file).get(f"{dataset}_{task}", {})
    rows = []

    def add(name, files, key="test", lat_key=None, params=None, k=None, own_cost=False):
        if not files:
            return
        r = _metric_rows(files, key)
        L = load_json(files[0]) if own_cost else lat.get(lat_key or name, {})
        fid = [f["spearman"] for f in (load_json(p).get("fidelity") for p in files) if f]
        rows.append({"model": name, "runs": len(files), "k": k,
                     "macro_f1": _pm(r["macro_f1"]), "accuracy": _pm(r["accuracy"]), "mcc": _pm(r["mcc"], 1.0, 4),
                     "shap_fidelity": round(float(np.mean(fid)), 3) if fid else None,
                     "params": params or L.get("params"), "size_kb": L.get("size_kb"),
                     "cpu_latency_b1_ms": L.get("latency_b1_ms"), "cpu_flows_per_s": L.get("throughput_b256")})

    tfiles = sorted((paths.outputs / "teacher").glob(f"{dataset}_{task}_{teacher_model}_s*/metrics.json"))
    add("teacher", tfiles, params=load_json(tfiles[0])["params"] if tfiles else None)
    add("xgboost", sorted((paths.outputs / "xgboost").glob(f"{dataset}_{task}_s*/metrics.json")))
    for kind in ("dtree", "logreg"):
        add(kind, sorted((paths.outputs / "baselines").glob(f"{dataset}_{task}_{kind}/metrics.json")), own_cost=True)
    kd_root = paths.outputs / "kd" / f"{dataset}_{task}"
    for v in variants:
        files = sorted(kd_root.glob(f"{v}_s*/metrics.json"))
        k = load_json(files[0])["k"] if files else None
        add(v, files, k=k)
        if v == "shield":
            add("shield_int8", files, key="test_int8", lat_key="shield_int8", k=k)
    return pd.DataFrame(rows)


def k_sweep_table(paths: Paths, dataset: str, task: str) -> pd.DataFrame:
    rows = []
    for f in sorted((paths.outputs / "kd" / f"{dataset}_{task}").glob("shield_k*_s*/metrics.json")):
        m = load_json(f)
        rows.append({"k": m["k"], "seed": m["seed"], "macro_f1": m["test"]["macro_f1"],
                     "macro_f1_int8": m["test_int8"]["macro_f1"],
                     "fidelity": (m.get("fidelity") or {}).get("spearman")})
    return pd.DataFrame(rows)


def kd_rank_table(paths: Paths, datasets_tasks: list[tuple[str, str]], variants: list[str]) -> pd.DataFrame:
    """Rows = (dataset, task, seed) blocks, columns = KD variants, values = test macro-F1 (for CD diagrams)."""
    rows = {}
    for ds, task in datasets_tasks:
        for v in variants:
            for f in (paths.outputs / "kd" / f"{ds}_{task}").glob(f"{v}_s*/metrics.json"):
                m = load_json(f)
                rows.setdefault((ds, task, m["seed"]), {})[v] = m["test"]["macro_f1"]
                if v == "shield":
                    rows[(ds, task, m["seed"])]["shield_int8"] = m["test_int8"]["macro_f1"]
    return pd.DataFrame.from_dict(rows, orient="index")


def inflation_table(paths: Paths, pairs: list[tuple[str, str, str]], teacher_model: str) -> pd.DataFrame:
    """(strict dataset, standard dataset, task) -> macro-F1 of seed-0 teacher and SHIELD under both protocols."""
    rows = []
    for strict, std, task in pairs:
        for model, rel in (("teacher", f"teacher/{{ds}}_{task}_{teacher_model}_s0/metrics.json"),
                           ("shield", f"kd/{{ds}}_{task}/shield_s0/metrics.json")):
            fs, fd = paths.outputs / rel.format(ds=strict), paths.outputs / rel.format(ds=std)
            if fs.exists() and fd.exists():
                a, b = load_json(fs)["test"]["macro_f1"], load_json(fd)["test"]["macro_f1"]
                rows.append({"dataset": strict, "task": task, "model": model, "strict": a, "standard": b,
                             "inflation_points": round(100 * (b - a), 2)})
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, out: Path, caption: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    with open(out.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, caption=caption, na_rep="--", float_format="%.4g"))
