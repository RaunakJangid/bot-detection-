"""Statistics over placement runs: summaries, paired Wilcoxon (Holm-corrected), Friedman ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

INSTANCE = ["topology", "k", "rho"]


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def summarise(df: pd.DataFrame, optimum: pd.DataFrame | None = None) -> pd.DataFrame:
    g = df.groupby(INSTANCE + ["algorithm"])
    out = g.agg(F_mean=("F", "mean"), F_std=("F", "std"), F_median=("F", "median"), F_min=("F", "min"),
                avg_ms=("avg", "mean"), fail_ms=("fail", "mean"), pct_sla=("pct_sla", "mean"),
                fail_pct_sla=("fail_pct_sla", "mean"), seconds=("seconds", "mean"), runs=("F", "size")).reset_index()
    if optimum is not None and len(optimum):
        out = out.merge(optimum[INSTANCE + ["F_opt"]], on=INSTANCE, how="left")
        out["gap_pct"] = 100 * (out["F_mean"] - out["F_opt"]) / out["F_opt"]
    return out


def wilcoxon_vs(df: pd.DataFrame, ours: str, others: list[str]) -> pd.DataFrame:
    """Paired by run seed; one-sided H1: ours has lower F. Holm-corrected within each instance."""
    rows = []
    for key, inst in df.groupby(INSTANCE):
        ref = inst[inst.algorithm == ours].set_index("seed")["F"]
        block = []
        for alg in others:
            other = inst[inst.algorithm == alg].set_index("seed")["F"]
            common = ref.index.intersection(other.index)
            if len(common) < 2:
                continue
            a, b = ref.loc[common].to_numpy(), other.loc[common].to_numpy()
            diff = a - b
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = float(wilcoxon(a, b, alternative="less", zero_method="zsplit").pvalue)
            wins, ties = int((diff < -1e-12).sum()), int((np.abs(diff) <= 1e-12).sum())
            block.append({**dict(zip(INSTANCE, key)), "vs": alg, "p": p, "wins": wins, "ties": ties,
                          "losses": len(common) - wins - ties, "median_diff": float(np.median(diff))})
        if block:
            for row, adj in zip(block, holm([r["p"] for r in block])):
                row["p_holm"] = adj
            rows.extend(block)
    return pd.DataFrame(rows)


def friedman_ranks(summary: pd.DataFrame, algorithms: list[str]) -> tuple[pd.Series, float | None]:
    """Average rank (1 = best mean F) across instances, and the Friedman test p-value."""
    piv = summary.pivot_table(index=INSTANCE, columns="algorithm", values="F_mean")
    piv = piv[[a for a in algorithms if a in piv.columns]].dropna()
    if piv.empty:
        return pd.Series(dtype=float), None
    ranks = piv.rank(axis=1).mean().sort_values()
    p = None
    if piv.shape[1] >= 3 and piv.shape[0] >= 2:
        p = float(friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue)
    return ranks, p
