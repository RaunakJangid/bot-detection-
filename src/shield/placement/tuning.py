"""Fair automatic tuning of the placement metaheuristics.

Every algorithm gets the same number of candidate settings (its defaults + random draws from its search
space), each run on the same TUNING instances: synthetic graphs drawn with different seeds from the
evaluation graphs, so evaluation instances are never seen. A candidate's score is its mean of
F / (best F any algorithm found on that instance); the lowest score wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def candidates(space: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Defaults ({}), then up to n-1 distinct random draws from the space."""
    out, seen = [{}], {()}
    keys = sorted(space)
    for _ in range(50 * n):
        if len(out) >= n:
            break
        c = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        sig = tuple(sorted(c.items()))
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def tuning_instances(tcfg: dict) -> list[tuple[str, int]]:
    return [(f"syn{n}s{tcfg['seed_offset'] + n}", int(k)) for n in tcfg["synthetic"] for k in tcfg["k_values"]]


def score(rows: list[dict]) -> pd.DataFrame:
    """rows: algorithm, candidate, topology, k, run, F -> one row per (algorithm, candidate) with its score."""
    df = pd.DataFrame(rows)
    best = df.groupby(["topology", "k"])["F"].transform("min")
    df["ratio"] = df["F"] / best
    return (df.groupby(["algorithm", "candidate"])["ratio"].mean().rename("score").reset_index())


def choose(scores: pd.DataFrame, cands: dict[str, list[dict]], inherit: dict[str, str]) -> dict:
    out = {}
    for alg, g in scores.groupby("algorithm"):
        best = g.loc[g["score"].idxmin()]
        default = g[g["candidate"] == 0]["score"]
        out[alg] = {"params": cands[alg][int(best["candidate"])], "score": float(best["score"]),
                    "default_score": float(default.iloc[0]) if len(default) else None,
                    "candidates": int(len(g))}
    for alg, parent in inherit.items():
        if parent in out:
            out[alg] = {**out[parent], "inherited_from": parent}
    return out
