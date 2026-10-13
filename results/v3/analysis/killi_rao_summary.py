"""Summary of the faithful Killi & Rao FFCCP benchmark (scripts/57_killi_rao.py, row 28af)."""

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = r"D:\shield_run\outputs_v3\killi_rao"
r = pd.read_json(O + r"\runs.jsonl", lines=True)
opt = json.load(open(O + r"\optimum.json"))
r["key"] = r.topology + "|" + r.K.astype(str) + "|" + r.p.astype(str)
proved = {k for k, v in opt.items() if v and v["status"] == 0}
print("instances", r.key.nunique(), "| optimum proved", len(proved), "| time-limited",
      sum(1 for v in opt.values() if v and v["status"] != 0))
best = r.groupby("key").z.min()
ref = {k: opt[k]["z"] if k in proved else min(best[k], opt[k]["z"] if opt.get(k) else np.inf) for k in best.index}
print("metaheuristic below a proven optimum (should be none):",
      [k for k in proved if k in best and best[k] < opt[k]["z"] - 1e-6])
r["ref"] = r.key.map(ref)
r["gap"] = 100 * (r.z / r.ref - 1)
r["hit"] = r.gap < 1e-6
r["proved"] = r.key.isin(proved)
s = r.groupby(["proved", "alg"]).agg(mean_gap=("gap", "mean"), worst=("gap", "max"), hit=("hit", "mean"),
                                     off5=("gap", lambda g: (g > 5).mean()), sec=("seconds", "mean")).round(3)
print(s.to_string())
m = r.groupby(["key", "alg"]).z.mean().unstack()
for b in ["gwo_sa", "sa", "ga", "gwo", "pso", "hybrid_linear", "random"]:
    d = m["hybrid_eho_aco"] - m[b]
    nz = d[d.abs() > 1e-9]
    p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
    print(f"hybrid vs {b:13s} better {int((d < -1e-9).sum()):2d} worse {int((d > 1e-9).sum()):2d} "
          f"tie {int((d.abs() <= 1e-9).sum()):2d} p={p:.3g}")
print("rank (mean z per instance):", m.rank(axis=1).mean().sort_values().round(2).to_dict())
tl = [(k, round(opt[k]["z"], 2), round(best[k], 2)) for k in opt if opt[k] and opt[k]["status"] != 0]
print("time-limited MILP incumbent vs best metaheuristic:", tl)
sec = [v["seconds"] for v in opt.values() if v and v["status"] == 0]
print("MILP seconds (proved): median %.1f max %.0f" % (np.median(sec), max(sec)))
for t in ["AttMpls", "Geant2012", "Surfnet"]:
    for K in (2, 3):
        ks = sorted([k for k in opt if k.startswith(f"{t}|{K}|")], key=lambda k: int(k.split("|")[2]))
        print(t, "K=%d" % K, [(int(k.split("|")[2]), round(ref[k], 2)) for k in ks])
