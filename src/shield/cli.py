"""Shared argument handling for scripts/*.py."""

from __future__ import annotations

import argparse

from shield.data.registry import Registry
from shield.utils.config import Paths


def parser(description: str, datasets: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (separate data/output roots)")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is present")
    p.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    if datasets:
        p.add_argument("--dataset", default="all",
                       help="all | main | cross | standard | a dataset name (configs/datasets.yaml) | comma list")
        p.add_argument("--task", default=None, help="restrict to one task (e.g. binary)")
    return p


def registry(args) -> Registry:
    return Registry(smoke=args.smoke)


def selected_datasets(args, stage: str | None = None) -> list[str]:
    reg = registry(args)
    names = reg.resolve(args.dataset)
    return reg.for_stage(stage, names) if stage else names


def paths_for(args) -> Paths:
    return Paths(smoke=args.smoke)
