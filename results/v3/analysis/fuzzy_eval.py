import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = ["hybrid_eho_aco", "hybrid_fuzzy", "hybrid_linear", "gwo_sa", "sa"]
rows = [json.loads(l) for l in open(r"D:\shield_run\outputs_v3\placement\runs.jsonl")]
d = pd.DataFrame([r for r in rows if r["algorithm"] in A and not r.get("skipped")])
s = pd.read_csv(r"D:\shield_run\outputs_v3\placement\summary.csv")
g = s[s.algorithm.isin(A)].dropna(subset=["gap_pct"])
print("mean gap % (37 exact):", g.groupby("algorithm").gap_pct.mean().round(3).to_dict())
blk = d.groupby(["topology", "algorithm"]).F.mean().unstack()[A]
print("topology ranks:", blk.rank(axis=1).mean().round(2).to_dict())
inst = d.groupby(["topology", "k", "rho", "algorithm"]).F.mean().unstack()
for a, b in (("hybrid_linear", "hybrid_eho_aco"), ("hybrid_linear", "gwo_sa"), ("hybrid_fuzzy", "hybrid_eho_aco"),
             ("hybrid_fuzzy", "gwo_sa")):
    x, t = inst[a] - inst[b], blk[a] - blk[b]
    print(f"{a} vs {b}: instances better {(x < -1e-9).sum()} worse {(x > 1e-9).sum()} "
          f"p={wilcoxon(x, zero_method='zsplit').pvalue:.2g} | topologies better {(t < 0).sum()}/8 p={wilcoxon(t).pvalue:.2g}")
h = d[d.history.map(len) > 0]
t99 = h.groupby("algorithm").history.apply(
    lambda c: np.mean([np.argmax(np.asarray(v) <= v[-1] * 1.001) / (len(v) - 1) for v in c]))
print("t99 (budget fraction to reach final):", t99.round(3).to_dict())
print("evals used (mean):", d.groupby("algorithm").n_evals.mean().round(0).to_dict())
