"""Validation-only selection of the distillation objective and the per-task feature/size budget.

Step 1  objective: plain KD (T, alpha) vs Decoupled KD (T, dkd_beta), all features, scored by the mean
        best validation macro-F1 over the main tasks (one global choice).
Step 2  teacher assistant: the step-1 winner distilled from a mid-size assistant instead of the teacher.
Step 3  features: ranking (global vs class-balanced SHAP, one global choice) x k x student size; per task,
        the cheapest (fewest parameters) setting within `tolerance` of that task's best validation score.
Test data is never read. Result: outputs/tuning/kd_choice.json, used by every later student run.
"""

from __future__ import annotations

import itertools

import numpy as np

ROOT = "kd_tune"


def objective_grid(tcfg: dict) -> list[dict]:
    grid = []
    for T, a in itertools.product(tcfg["objective"]["kd"]["T"], tcfg["objective"]["kd"]["alpha"]):
        grid.append({"kd": "kd", "T": float(T), "alpha": float(a), "beta": round(1.0 - float(a), 6)})
    for T, b in itertools.product(tcfg["objective"]["dkd"]["T"], tcfg["objective"]["dkd"]["dkd_beta"]):
        grid.append({"kd": "dkd", "T": float(T), "alpha": 1.0, "beta": 1.0, "dkd_alpha": 1.0, "dkd_beta": float(b)})
    return grid


def objective_name(o: dict) -> str:
    tail = f"a{o['alpha']}" if o["kd"] == "kd" else f"b{o['dkd_beta']}"
    return f"obj_{o['kd']}_T{o['T']}_{tail}" + ("_ta" if o.get("teacher") == "assistant" else "")


def feature_grid(tcfg: dict) -> list[dict]:
    f = tcfg["features"]
    return [{"ranking": r, "k": int(k), "hidden": list(h)}
            for r, k, h in itertools.product(f["ranking"], f["k"], f["hidden"])]


def feature_name(c: dict) -> str:
    return f"feat_{'cb' if c['ranking'] == 'class_balanced' else 'gl'}_k{c['k']}_h{'-'.join(map(str, c['hidden']))}"


def mean_val(scores: dict[str, dict[str, float]], name: str) -> float:
    return float(np.mean(list(scores[name].values())))


def choose_objective(scores: dict[str, dict[str, float]], grid: list[dict]) -> dict:
    best = max(grid, key=lambda o: mean_val(scores, objective_name(o)))
    return dict(best)


def choose_features(scores: dict[str, dict[str, float]], params: dict[str, dict[str, int]], grid: list[dict],
                    tasks: list[str], tolerance: float) -> tuple[str, dict[str, dict]]:
    """Global ranking method by mean score over all grid points and tasks; then per task the cheapest
    (k, hidden) within `tolerance` of the best score for that ranking."""
    rankings = sorted({c["ranking"] for c in grid})
    ranking = max(rankings, key=lambda r: np.mean([scores[feature_name(c)][t] for c in grid if c["ranking"] == r
                                                    for t in tasks]))
    cands = [c for c in grid if c["ranking"] == ranking]
    per_task = {}
    for t in tasks:
        best = max(scores[feature_name(c)][t] for c in cands)
        ok = [c for c in cands if scores[feature_name(c)][t] >= best - tolerance]
        pick = min(ok, key=lambda c: (params[feature_name(c)][t], c["k"]))
        per_task[t] = {"k": pick["k"], "hidden": pick["hidden"]}
    return ranking, per_task
