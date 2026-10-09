"""Compute cost of the placement methods on this machine (i7-12700F, one process per run, 20,000 evaluations):
mean wall-clock seconds per run, evaluations per second, and the share of the budget needed to get within 1% of
the exact optimum (instances with a known optimum; runs that never get there count as 1.0)."""
import json

import numpy as np
import pandas as pd

ALG = ["gwo_sa", "hybrid_linear", "hybrid_eho_aco", "sa", "pso", "hho", "de", "gwo", "ga", "aco", "eho"]
rows = [json.loads(l) for l in open(r"D:\shield_run\outputs_v3\placement\runs.jsonl")]
d = pd.DataFrame([r for r in rows if r["algorithm"] in ALG and not r.get("skipped")])
opt = pd.DataFrame([json.loads(l) for l in open(r"D:\shield_run\outputs_v3\placement\optimum.jsonl")])
opt = opt[~opt.skipped.astype(bool)].set_index(["topology", "k", "rho"])["F_opt"]
d = d.join(opt, on=["topology", "k", "rho"])


def t1(r):
    if np.isnan(r.F_opt) or not r.history:
        return np.nan
    h = np.asarray(r.history, float)
    ok = np.flatnonzero(h <= 1.01 * r.F_opt)
    return ok[0] / (len(h) - 1) if len(ok) else 1.0


d["t1"] = d.apply(t1, axis=1)
g = d.groupby("algorithm").agg(seconds=("seconds", "mean"), evals=("n_evals", "mean"), t1=("t1", "mean"),
                               reached=("t1", lambda s: float((s.dropna() < 1).mean())))
g["evals_per_s"] = g.evals / g.seconds
g["seconds_to_1pct"] = g.seconds * g.t1
print(g[["seconds", "evals_per_s", "t1", "seconds_to_1pct", "reached"]].sort_values("seconds_to_1pct").round(3).to_string())
