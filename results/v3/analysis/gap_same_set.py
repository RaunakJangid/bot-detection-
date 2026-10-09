import math
import pandas as pd

s3 = pd.read_csv(r"D:\shield_run\outputs_v3\placement\summary.csv").dropna(subset=["gap_pct"])
s2 = pd.read_csv(r"D:\shield_run\outputs_v2\placement\summary.csv").dropna(subset=["gap_pct"])
small = s3[[math.comb(int(n), int(k)) <= 2_000_000 for n, k in zip(s3.n, s3.k)]]
print("v2 small instances:", s2.groupby(["topology", "k", "rho"]).ngroups, "| v3 same-rule small:",
      small.groupby(["topology", "k", "rho"]).ngroups)
for name, d in (("v2 (small)", s2), ("v3 (same small set)", small), ("v3 (all 37 exact)", s3)):
    g = d.groupby("algorithm")["gap_pct"].mean()
    print(name, {a: round(g.get(a, float("nan")), 3) for a in
                 ["hybrid_eho_aco", "gwo_sa", "sa", "pso", "gwo", "aco", "ga", "foa", "hho", "de", "woa", "eho"]})
med = s3[[math.comb(int(n), int(k)) > 2_000_000 for n, k in zip(s3.n, s3.k)]]
print("v3 medium/large exact instances:", med.groupby(["topology", "k", "rho"]).ngroups,
      med.groupby("algorithm")["gap_pct"].mean().sort_values().round(2).head(6).to_dict())
