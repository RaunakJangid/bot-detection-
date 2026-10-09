import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ALG = ["hybrid_eho_aco", "hybrid_no_ants", "hybrid_no_ls", "eho", "aco", "sa", "gwo", "gwo_sa", "pso", "de", "hho"]
rows = [json.loads(l) for l in open(r"D:\shield_run\outputs_v3\placement\runs.jsonl")]
d = pd.DataFrame([r for r in rows if r["algorithm"] in ALG and not r.get("skipped")])

# 8: topology as the experimental unit (mean over k, rho, seeds), and per-instance medians
blk = d.groupby(["topology", "algorithm"])["F"].mean().unstack()[ALG]
print("ranks over 8 topology blocks:", blk.rank(axis=1).mean().sort_values().round(2).to_dict())
print("Friedman p (8 blocks):", friedmanchisquare(*[blk[a] for a in ALG]).pvalue)
x = blk["hybrid_eho_aco"] - blk["gwo_sa"]
print("hybrid - gwo_sa per topology:", x.round(4).to_dict(), "| wilcoxon p", round(wilcoxon(x).pvalue, 3))

# 10: convergence: evaluations (of 20000) to reach within 0.1% of the run's final best; late improvement
def conv(h):
    h = np.asarray(h, float)
    f = h[-1]
    return np.argmax(h <= f * 1.001) / (len(h) - 1), (h[len(h) // 2] - f) / f
cv = d[d.history.map(len) > 0].assign(**{"t99": lambda t: t.history.map(lambda h: conv(h)[0]),
                                          "late": lambda t: t.history.map(lambda h: conv(h)[1])})
print("\nfraction of budget to reach final-0.1% | % improvement in 2nd half of budget")
print(cv.groupby("algorithm")[["t99", "late"]].mean().mul([1, 100]).round(3).sort_values("t99").to_string())

# 6: objective structure: weighted, normalised term shares of the final solutions (weights from placement.yaml)
w = {"avg": 1.0, "max": 0.5, "inter": 0.1, "imbalance": 0.2, "fail": 1.0}
best = d[d.algorithm == "gwo_sa"]
raw = best[list(w)].abs()
print("\nraw term means for gwo_sa solutions (ms / CoV):", raw.mean().round(3).to_dict())
print("overload>0 share of runs:", float((best["overload"] > 1e-9).mean()))
