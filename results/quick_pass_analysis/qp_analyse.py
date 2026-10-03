"""Quick-pass checks (a)-(j) and the Phase-2 step-4 gamma rule. Read-only over D:/shield_run/outputs."""
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

O = Path(r"D:/shield_run/outputs")
D = Path(r"D:/shield_run/data")
MAIN = {"ciciot": ["binary", "family", "class34"], "ciciomt": ["binary", "category", "attack"],
        "botiot": ["binary", "category"]}
LEAKY = {"ciciot": "IAT", "ciciomt": "IAT", "botiot": "seq"}
out = {}


def j(p):
    return json.load(open(p, encoding="utf-8"))


# (a) data
a = {}
for f in sorted(O.glob("data_report_*.json")):
    r = j(f)
    ds = r["dataset"]
    cc = r.get("class_counts", {})
    missing = {}
    for task, splits in cc.items():
        tr = splits.get("train", {})
        miss = [c for c, n in tr.items() if n == 0]
        names = r["tasks"].get(task, [])
        miss += [c for c in names if c not in tr]
        if miss:
            missing[task] = sorted(set(miss))
    a[ds] = {"rows": r["n_rows"], "n_features": r["n_features"], "dropped": r["dropped_features"],
             "dedup": {k: r["dedup"].get(k) for k in ("rows_in", "duplicate_rows_dropped", "conflicting_hashes",
                                                      "conflicting_rows_dropped", "rows_out")},
             "missing_train_classes": missing}
out["a"] = a

# (b) leakage audit
lk = j(O / "leakage" / "summary.json")
b = {}
for key, v in lk.items():
    src = key.split("_std")[0]
    allf = v["all_features_macro_f1"]
    csv = pd.read_csv(O / "leakage" / f"{key}.csv")
    fcol = [c for c in csv.columns if "feature" in c.lower()][0]
    scol = [c for c in csv.columns if "alone" in c.lower() or "macro" in c.lower()][0]
    csv = csv.sort_values(scol, ascending=False).reset_index(drop=True)
    leak = LEAKY[src]
    rank = int(csv.index[csv[fcol] == leak][0]) + 1 if (csv[fcol] == leak).any() else None
    suspects = csv[(csv[scol] >= 0.8 * allf) & (csv[fcol] != leak)][[fcol, scol]].values.tolist()
    b[key] = {"all_features": allf, "chance": v["chance"], "leaky": leak, "leaky_rank": rank,
              "leaky_alone": float(csv.loc[csv[fcol] == leak, scol].iloc[0]) if rank else None,
              "top5": csv.head(5)[[fcol, scol]].values.tolist(), "other_suspects_ge80pct": suspects}
out["b"] = b

# (c) inflation
out["c"] = pd.read_csv(O / "paper" / "tables" / "leakage_inflation.csv").to_dict("records")

# (d) teachers vs xgboost, early stop, strict SHAP features
d = {}
for ds, tasks in MAIN.items():
    for t in tasks:
        tm = j(O / "teacher" / f"{ds}_{t}_resmlp_s0" / "metrics.json")
        h = tm["history"]
        best = max(h, key=lambda e: e["val_macro_f1"])
        xg = j(O / "xgboost" / f"{ds}_{t}_s0" / "metrics.json") if (O / "xgboost" / f"{ds}_{t}_s0").exists() else None
        imp = j(O / "teacher" / f"{ds}_{t}_resmlp_s0" / "shap" / "importance.json")
        feats = json.dumps(imp)
        d[f"{ds}_{t}"] = {"teacher": tm["test"]["macro_f1"], "xgb": xg["test"]["macro_f1"] if xg else None,
                          "last_epoch": h[-1]["epoch"], "best_epoch": best["epoch"],
                          "leaky_in_shap": f'"{LEAKY[ds]}"' in feats}
out["d"] = d

# (e) KD ablation from detection tables (main tasks)
e = {}
for ds, tasks in MAIN.items():
    for t in tasks:
        tab = pd.read_csv(O / "paper" / "tables" / f"detection_{ds}_{t}.csv").set_index("model")
        e[f"{ds}_{t}"] = tab[["macro_f1", "shap_fidelity", "params", "k"]].to_dict("index")
out["e"] = e

# (f) latency
out["f"] = j(O / "latency" / "latency.json")

# (g) placement
summ = pd.read_csv(O / "placement" / "summary.csv")
g = {"friedman": j(O / "placement" / "friedman.json")}
gap = summ.dropna(subset=["gap_pct"])
g["gap_by_alg"] = gap.groupby("algorithm")["gap_pct"].agg(["mean", "median", "max"]).round(3).to_dict("index")
g["hybrid_zero_gap_share"] = float((gap[gap.algorithm == "hybrid_eho_aco"]["gap_pct"] < 1e-6).mean())
ranks = {}
for topo in ["Cogentco", "Kdl", "syn500"]:
    s = summ[summ.topology == topo].copy()
    s["r"] = s.groupby(["k", "rho"])["F_mean"].rank()
    ranks[topo] = s.groupby("algorithm")["r"].mean().sort_values().round(2).to_dict()
g["large_topology_avg_rank"] = ranks
g["runtime_s_mean_by_alg"] = summ.groupby("algorithm")["seconds"].mean().round(2).to_dict()
out["g"] = g

# (h) coupled
out["h"] = pd.read_csv(O / "coupled" / "min_controllers.csv").to_dict("records")

# (i) cross
fs = pd.read_csv(O / "cross" / "fewshot_summary.csv")
piv = fs.pivot_table(index=["source", "target", "task", "model", "fraction"], columns="kind", values="mean").reset_index()
piv["ft_minus_scratch"] = piv["finetuned"] - piv["scratch"]
out["i_fewshot"] = piv.round(4).to_dict("records")
zs = pd.read_csv(O / "cross" / "zero_shot_summary.csv", header=[0, 1], index_col=[0, 1, 2, 3, 4])
zs.columns = ["_".join(c) for c in zs.columns]
out["i_zero_shot"] = zs.reset_index().round(4).to_dict("records")
out["i_unseen"] = pd.read_csv(O / "cross" / "unseen_summary.csv").round(4).to_dict("records")
sa = j(O / "cross" / "shap_agreement.json")
out["i_shap_agreement"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("top10_x")} for k, v in sa.items()}

# (j) expected outputs
sys.path.insert(0, r"C:/Users/Prakash/shield/scripts")
os.chdir(r"C:/Users/Prakash/shield")
from shield.utils.config import Paths  # noqa: E402
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("run_all", r"C:/Users/Prakash/shield/scripts/run_all.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
missing = [p for p in ra.expected(Paths()) if not glob.glob(p)]
out["j"] = {"missing_expected": missing, "n_tables": len(list((O / "paper" / "tables").glob("*"))),
            "n_figures": len(list((O / "paper" / "figures").glob("*")))}

# Step 4: gamma rule on VALIDATION only
s4 = {}
for ds, tasks in MAIN.items():
    for t in tasks:
        def best_val(v):
            m = j(O / "kd" / f"{ds}_{t}" / f"{v}_s0" / "metrics.json")
            return max(e["val_macro_f1"] for e in m["history"])
        s4[f"{ds}_{t}"] = {"shield": best_val("shield"), "kd_shap": best_val("kd_shap")}
n_below = sum(v["shield"] < v["kd_shap"] for v in s4.values())
out["step4"] = {"per_task": s4, "n_shield_below_kd_shap": n_below, "sweep_triggered": n_below >= 5}

json.dump(out, open(Path(__file__).with_name("qp_results.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
