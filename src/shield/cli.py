"""Shared argument handling for scripts/*.py."""

from __future__ import annotations

import argparse

from shield.utils.config import Paths


def parser(description: str, datasets: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (separate data/output roots)")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is present")
    p.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    if datasets:
        p.add_argument("--dataset", choices=["all", "ciciot", "botiot"], default="all")
        p.add_argument("--task", default=None, help="restrict to one task (e.g. binary)")
    return p


def selected_datasets(args) -> list[str]:
    return ["ciciot", "botiot"] if args.dataset == "all" else [args.dataset]


def tasks_for(cfg_tasks: dict, dataset: str, only: str | None) -> list[str]:
    tasks = cfg_tasks[dataset]
    return [t for t in tasks if only is None or t == only]


def paths_for(args) -> Paths:
    return Paths(smoke=args.smoke)
