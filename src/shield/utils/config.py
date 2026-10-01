"""YAML config loading with a `smoke:` override section and project-relative paths."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(name: str, smoke: bool = False) -> dict[str, Any]:
    """Load configs/<name>.yaml; with smoke=True the `smoke:` section is merged on top."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    smoke_cfg = cfg.pop("smoke", {}) or {}
    return deep_merge(cfg, smoke_cfg) if smoke else cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Paths:
    """Resolved project locations. Smoke runs use separate data/output roots."""

    def __init__(self, smoke: bool = False):
        cfg = load_config("paths", smoke)
        self.smoke = smoke
        self.ciciot_zip = Path(cfg["ciciot_zip"])
        self.botiot_zip = Path(cfg["botiot_zip"])
        self.ciciomt_zip = Path(cfg["ciciomt_zip"])
        self.data = resolve(cfg["data_dir"])
        self.outputs = resolve(cfg["outputs_dir"])

    def interim(self, dataset: str) -> Path:
        return self.data / "interim" / dataset

    def processed(self, dataset: str) -> Path:
        return self.data / "processed" / dataset

    @property
    def topologies(self) -> Path:
        # Topologies are small and identical for smoke and full runs.
        return resolve("data") / "topologies"

    def out(self, *parts: str) -> Path:
        p = self.outputs.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p
