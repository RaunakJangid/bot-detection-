import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

V3 = Path(r"D:\shield_run\outputs_v3")
pd.set_option("display.width", 220, "display.max_columns", 30)


def wx(d):
    d = np.asarray(d)
    return float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0


print("=== 1. v3 students (ensemble teacher) vs v2 students, test macro-F1 %")
ph = pd.read_csv(V3 / "posthoc" / "summary.csv")
ph["tk"] = ph.dataset + "_" + ph.task
for new, old in (("v3-kd_all_ens", "kd_all"), ("v3-kd_shap_ens", "kd_shap"), ("v3-kd_shap_ens", "shield")):
    for col in ("test_raw", "test_bias"):
        a = ph[ph.model == new].set_index(["tk", "seed"])[col]
        b = ph[ph.model == old].set_index(["tk", "seed"])[col]
        d = (a - b).dropna()
        print(f"{new:15s} vs {old:8s} [{col:9s}] mean {100*d.mean():+.2f} pts, wins {(d>0).sum()}/{len(d)}, p={wx(d):.2g}")
tab = (ph[ph.model.isin(["teacher", "kd_all", "kd_shap", "shield", "v3-kd_all_ens", "v3-kd_shap_ens"])]
       .groupby(["tk", "model"])[["test_raw", "test_bias"]].mean().mul(100).round(2).unstack("model"))
print(tab["test_bias"])

print("\n=== 2. cascade (rules), matched budget (val), mean over seeds")
cs = pd.read_csv(V3 / "cascade" / "rules" / "summary.csv")
m = cs[cs.matched]
g = m.groupby(["dataset", "task", "student", "expert"]).agg(student0=("test_macro_f1", "size"),
    budget=("budget", "mean"), esc=("test_escalated", "mean"), f1=("test_macro_f1", "mean"), expert=("expert_test_f1", "mean"))
base = cs[cs.budget == 0].groupby(["dataset", "task", "student", "expert"])["test_macro_f1"].mean()
g["student_alone"] = base
g = g.drop(columns="student0")
print((g[["student_alone", "f1", "expert"]] * 100).round(2).join((g[["budget", "esc"]] * 100).round(1)).to_string())
for b in (0.02, 0.05, 0.1):
    x = cs[(cs.budget == b) & (cs.student == "shield") & (cs.expert == "ensemble")]
    print(f"shield->ensemble at {b:.0%}: mean test esc {100*x.test_escalated.mean():.1f}%, "
          f"gap to ensemble {100*(x.expert_test_f1 - x.test_macro_f1).mean():.2f} pts")

print("\n=== 3. int8")
i8 = pd.read_csv(V3 / "int8" / "summary.csv")
print(i8.groupby(["dataset", "task"])[["drop_pts", "drop_pts_w8a16", "drop_pts_v2"]].mean().round(2).to_string())
print("modes:", i8["mode"].value_counts().to_dict(), "| overall mean drop W8A8 %.2f, W8A16 %.2f, v2 %.2f" %
      (i8.drop_pts.mean(), i8.drop_pts_w8a16.mean(), i8.drop_pts_v2.mean()))
print("bytes (shield, per task):", i8[i8.variant == "shield"].groupby(["dataset", "task"])[["int8_bytes", "fp32_bytes"]].first().to_dict("index"))

print("\n=== 4. explanation fidelity")
print(json.dumps({k: v for k, v in json.load(open(V3 / "fidelity" / "wilcoxon.json")).items() if k != "per_task"}))
fd = pd.read_csv(V3 / "fidelity" / "summary.csv")
print(fd.pivot_table(index=["dataset", "task"], columns="variant", values="spearman").round(3).to_string())

print("\n=== 5. placement v3")
pl = V3 / "placement"
fr = json.load(open(pl / "friedman.json"))
print("friedman p", fr["friedman_p"], "CD", round(fr["nemenyi_cd"], 2), "instances", fr["instances"])
print({k: round(v, 2) for k, v in fr["average_rank"].items()})
s = pd.read_csv(pl / "summary.csv")
if "gap_pct" in s:
    gp = s.dropna(subset=["gap_pct"])
    print("instances with exact optimum:", gp.groupby(["topology", "k", "rho"]).ngroups,
          "topologies:", sorted(gp.topology.unique()))
    print(gp.groupby("algorithm")["gap_pct"].agg(["mean", "median", "max"]).sort_values("mean").round(3).head(14).to_string())
w = pd.read_csv(pl / "wilcoxon.csv")
print((w.assign(better=w.p_holm < 0.05, worse=w.p_worse_holm < 0.05).groupby("vs")[["better", "worse"]].sum()
       .sort_values("better")).to_string())
big = s[s.topology.isin(["Cogentco", "syn200", "syn500", "Kdl"])]
print("large-graph mean F:", big.groupby("algorithm")["F_mean"].mean().sort_values().round(4).head(10).to_dict())

print("\n=== 6. coupling (v3)")
print(pd.read_csv(V3 / "coupled" / "min_controllers.csv").to_string())
print(json.dumps(json.load(open(V3 / "coupled" / "coupled_meta.json"))["accuracy"], indent=0))
