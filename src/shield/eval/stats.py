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
                fail_pct_sla=("fail_pct_sla", "mean"), seconds=("seconds", "mean"), runs=("F", "size"),
                n=("n", "first")).reset_index()
    if optimum is not None and len(optimum):
        out = out.merge(optimum[INSTANCE + ["F_opt"]], on=INSTANCE, how="left")
        out["gap_pct"] = 100 * (out["F_mean"] - out["F_opt"]) / out["F_opt"]
    return out


def wilcoxon_vs(df: pd.DataFrame, ours: str, others: list[str]) -> pd.DataFrame:
    """Paired by run seed, both directions: p = one-sided H1 "ours has lower F" (better), p_worse = one-sided
    H1 "ours has higher F" (worse). Each is Holm-corrected within each instance, so a method can be reported
    as significantly better, significantly worse, or neither."""
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
                p = p_worse = 1.0
            else:
                p = float(wilcoxon(a, b, alternative="less", zero_method="zsplit").pvalue)
                p_worse = float(wilcoxon(a, b, alternative="greater", zero_method="zsplit").pvalue)
            wins, ties = int((diff < -1e-12).sum()), int((np.abs(diff) <= 1e-12).sum())
            block.append({**dict(zip(INSTANCE, key)), "vs": alg, "p": p, "p_worse": p_worse, "wins": wins,
                          "ties": ties, "losses": len(common) - wins - ties, "median_diff": float(np.median(diff)),
                          "mean_diff_pct": float(100 * (a.mean() - b.mean()) / max(abs(b.mean()), 1e-12))})
        if block:
            for row, adj, adj_w in zip(block, holm([r["p"] for r in block]), holm([r["p_worse"] for r in block])):
                row["p_holm"], row["p_worse_holm"] = adj, adj_w
            rows.extend(block)
    return pd.DataFrame(rows)


def friedman_ranks(summary: pd.DataFrame, algorithms: list[str]) -> tuple[pd.Series, float | None]:
    """Average rank (1 = best mean F) across instances, and the Friedman test p-value."""
    piv = summary.pivot_table(index=INSTANCE, columns="algorithm", values="F_mean")
    piv = piv[[a for a in algorithms if a in piv.columns]].dropna()
    return ranks_and_friedman(piv, higher_is_better=False)


# Nemenyi critical values q_0.05 (Demšar 2006, Table 5), by number of compared methods k.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
               11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def ranks_and_friedman(table: pd.DataFrame, higher_is_better: bool) -> tuple[pd.Series, float | None]:
    """table: rows = blocks (datasets/instances), columns = methods. Average rank (1 = best) + Friedman p."""
    table = table.dropna()
    if table.empty:
        return pd.Series(dtype=float), None
    ranks = table.rank(axis=1, ascending=not higher_is_better).mean().sort_values()
    p = None
    if table.shape[1] >= 3 and table.shape[0] >= 2:
        p = float(friedmanchisquare(*[table[c] for c in table.columns]).pvalue)
    return ranks, p


def nemenyi_cd(k: int, n_blocks: int) -> float | None:
    """Critical difference of average ranks at alpha = 0.05 (two methods differ if ranks differ by >= CD)."""
    q = NEMENYI_Q05.get(k)
    return None if q is None or n_blocks < 1 else q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
