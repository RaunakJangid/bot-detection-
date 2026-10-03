"""Helpers for the detector-capacity coupling experiment (scripts/07b_coupled.py)."""

from __future__ import annotations

import numpy as np

from shield.placement.baselines import greedy_kmedian


def topology_sla(D: np.ndarray, lam: np.ndarray, sla_cfg: dict, k_ref: int, absolute_ms: float) -> float:
    """Relative SLA: slack x the quantile of switch-to-nearest-controller propagation latency of the greedy
    k-median placement with k_ref controllers (capacity ignored), i.e. what a well-placed k_ref-controller
    network achieves on this topology. mode != "relative" returns the absolute limit."""
    if sla_cfg.get("mode", "absolute") != "relative":
        return float(absolute_ms)
    C = greedy_kmedian(D, lam, k_ref)
    return float(sla_cfg["slack"] * np.quantile(D[:, C].min(1), sla_cfg["quantile"]))


def feasible(d: dict, k: int, cfg: dict) -> bool:
    """SLA met by `sla_target` of switches with no overload, and (optionally) also after the worst
    single-controller failure."""
    ok = d["overload"] <= 1e-9 and d["pct_sla"] >= cfg["sla_target"]
    if cfg["require_failover"]:
        ok = ok and k > 1 and d["fail_overload"] <= 1e-9 and d["fail_pct_sla"] >= cfg["sla_target"]
    return ok
