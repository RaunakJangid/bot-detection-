"""Paper figures. Every figure is written as PDF and PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {"hybrid_eho_aco": "#d1495b", "eho": "#edae49", "aco": "#00798c", "ga": "#30638e",
           "pso": "#7b2d8e", "sa": "#66a182", "random": "#8d96a3", "kmeans": "#5c4d3c",
           "kmedian": "#2e4057", "kcenter": "#a3a380", "pagerank": "#c08497", "ilp_ckm": "#000000"}


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def pareto(lat_entry: dict, out: Path, title: str) -> None:
    pts = [(name, e["size_kb"], e["macro_f1"]) for name, e in lat_entry.items() if "macro_f1" in e]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, size, f1 in pts:
        color = "#d1495b" if name.startswith("shield") else ("#30638e" if name == "teacher" else "#8d96a3")
        ax.scatter(size, 100 * f1, color=color, s=40, zorder=3)
        ax.annotate(name, (size, 100 * f1), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("model size (KB, log scale)")
    ax.set_ylabel("test macro-F1 (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    save(fig, out)


def confusion_heatmap(cm: np.ndarray, classes: list[str], out: Path, title: str) -> None:
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(4, 0.35 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(norm, ax=ax, cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes,
                cbar_kws={"shrink": 0.7}, annot=len(classes) <= 10, fmt=".2f")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    save(fig, out)


def f1_vs_k(df: pd.DataFrame, out: Path, title: str, teacher_f1: float | None) -> None:
    if df.empty:
        return
    g = df.groupby("k")["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(g["k"], 100 * g["mean"], yerr=100 * g["std"].fillna(0), marker="o", color="#d1495b",
                capsize=3, label="SHIELD student")
    if teacher_f1 is not None:
        ax.axhline(100 * teacher_f1, color="#30638e", ls="--", label="teacher (all features)")
    ax.set_xlabel("selected features k"); ax.set_ylabel("test macro-F1 (%)"); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, out)


def objective_vs_k(summary: pd.DataFrame, rho: float, out_dir: Path) -> None:
    sub = summary[np.isclose(summary.rho, rho)]
    for topo, g in sub.groupby("topology"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for alg, h in g.groupby("algorithm"):
            h = h.sort_values("k")
            ax.errorbar(h["k"], h["F_mean"], yerr=h["F_std"].fillna(0), marker="o", capsize=2, label=alg,
                        color=PALETTE.get(alg), lw=2.2 if alg == "hybrid_eho_aco" else 1)
        ax.set_xlabel("controllers k"); ax.set_ylabel("objective F (lower is better)")
        ax.set_title(f"{topo}, rho={rho}"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        save(fig, out_dir / f"objective_vs_k_{topo}")


def convergence(runs: list[dict], topo: str, k: int, rho: float, budget: int, out: Path) -> None:
    sel = [r for r in runs if r["topology"] == topo and r["k"] == k and np.isclose(r["rho"], rho)
           and r.get("history")]
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for alg in sorted({r["algorithm"] for r in sel}):
        H = np.array([r["history"] for r in sel if r["algorithm"] == alg], dtype=float)
        x = np.linspace(1, budget, H.shape[1])
        ax.plot(x, np.nanmean(H, 0), label=alg, color=PALETTE.get(alg), lw=2.2 if alg == "hybrid_eho_aco" else 1)
    ax.set_xlabel("objective evaluations"); ax.set_ylabel("best F (mean over runs)")
    ax.set_title(f"Convergence: {topo}, k={k}, rho={rho}"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    save(fig, out)


def resilience_bars(df: pd.DataFrame, k: int, rho: float, metric: str, out: Path, ylabel: str) -> None:
    sub = df[(df.k == k) & np.isclose(df.rho, rho)]
    if sub.empty or metric not in sub:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(sub, x="topology", y=metric, hue="algorithm", ax=ax, palette=PALETTE, errorbar="sd")
    ax.set_ylabel(ylabel); ax.set_title(f"k={k}, rho={rho}")
    ax.legend(fontsize=7, ncol=3)
    save(fig, out)


def coupled_bars(mink: pd.DataFrame, out: Path, k_max: int) -> None:
    if mink.empty:
        return
    long = mink.reset_index().melt(id_vars=["topology", "rho"], var_name="model", value_name="k")
    long["k"] = long["k"].fillna(k_max + 1)
    long["case"] = long["topology"] + ", rho=" + long["rho"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(long, x="case", y="k", hue="model", ax=ax)
    ax.axhline(k_max + 1, color="grey", ls=":", lw=1)
    ax.set_ylabel(f"controllers needed (>{k_max} shown as {k_max + 1})"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    save(fig, out)
