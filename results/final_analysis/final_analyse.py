"""Full-run (Phase 3) numbers for FINAL_REPORT.md. Read-only over D:/shield_run/outputs."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = Path(r"D:/shield_run/outputs")
MAIN = {"ciciot": ["binary", "family", "class34"], "ciciomt": ["binary", "category", "attack"],
        "botiot": ["binary", "category"]}
VARIANTS = ["scratch_all", "kd_all", "kd_shap", "shield", "kd_random", "kd_mi", "kd_variance"]
out = {}


def j(p):
    return json.load(open(p, encoding="utf-8"))


def ms(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


# KD per task: test macro-F1, int8, fidelity, best val (per seed)
kd = {}
for ds, tasks in MAIN.items():
    for t in tasks:
        row = {}
        for v in VARIANTS:
            f, q, fid, val = [], [], [], []
            for p in sorted(glob.glob(str(O / "kd" / f"{ds}_{t}" / f"{v}_s*" / "metrics.json"))):
                m = j(p)
                f.append(m["test"]["macro_f1"])
                q.append(m["test_int8"]["macro_f1"])
                val.append(max(e["val_macro_f1"] for e in m["history"]))
                if m.get("fidelity"):
                    fid.append(m["fidelity"]["spearman"])
            row[v] = {"f1": ms(f), "int8": ms(q), "int8_drop": ms(np.array(f) - np.array(q)), "val": ms(val),
                      "fidelity": ms(fid) if fid else None, "per_seed_f1": [round(x, 4) for x in f]}
        tm = [j(p)["test"]["macro_f1"] for p in sorted(glob.glob(str(O / "teacher" / f"{ds}_{t}_resmlp_s*" / "metrics.json")))]
        row["teacher"] = {"f1": ms(tm)}
        for b in ("dtree", "logreg"):
            p = O / "baselines" / f"{ds}_{t}_{b}" / "metrics.json"
            row[b] = {"f1": ms([j(p)["test"]["macro_f1"]])} if p.exists() else None
        p = O / "xgboost" / f"{ds}_{t}_s0" / "metrics.json"
        row["xgboost"] = {"f1": ms([j(p)["test"]["macro_f1"]])} if p.exists() else None
        sh, ks = np.array(row["shield"]["per_seed_f1"]), np.array(row["kd_shap"]["per_seed_f1"])
        row["shield_minus_kd_shap"] = {"mean": float((sh - ks).mean()), "seeds_shield_better": int((sh > ks).sum())}
        kd[f"{ds}_{t}"] = row
out["kd"] = kd

# paired test shield vs kd_shap over all 40 (task, seed) blocks
a, b = [], []
for r in kd.values():
    a += r["shield"]["per_seed_f1"]
    b += r["kd_shap"]["per_seed_f1"]
out["shield_vs_kd_shap_wilcoxon"] = {"n": len(a), "p": float(wilcoxon(a, b).pvalue),
                                     "mean_diff_points": float(100 * (np.mean(a) - np.mean(b))),
                                     "blocks_shield_better": int(sum(x > y for x, y in zip(a, b)))}


def cd_groups(ranks: dict, cd: float):
    items = sorted(ranks.items(), key=lambda kv: kv[1])
    best = items[0][1]
    tied_with_best = [k for k, r in items if r - best < cd]
    cliques = []
    for i, (k, r) in enumerate(items):
        grp = [kk for kk, rr in items[i:] if rr - r < cd]
        if len(grp) > 1 and not any(set(grp) <= set(c) for c in cliques):
            cliques.append(grp)
    return {"order": items, "tied_with_best": tied_with_best, "cliques": cliques}


kr = j(O / "paper" / "tables" / "kd_ablation_ranks.json")
out["kd_cd"] = {"friedman_p": kr["friedman_p"], "cd": kr["nemenyi_cd"], "blocks": kr["blocks"],
                **cd_groups(kr["average_rank"], kr["nemenyi_cd"])}
pf = j(O / "placement" / "friedman.json")
out["placement_cd"] = {"friedman_p": pf["friedman_p"], "cd": pf["nemenyi_cd"], "instances": pf["instances"],
                       **cd_groups(pf["average_rank"], pf["nemenyi_cd"])}

# placement gap and large-topology ranks with 30 runs
summ = pd.read_csv(O / "placement" / "summary.csv")
gap = summ.dropna(subset=["gap_pct"])
out["placement_gap"] = gap.groupby("algorithm")["gap_pct"].agg(["mean", "median", "max"]).round(3).to_dict("index")
out["hybrid_zero_gap_share"] = float((gap[gap.algorithm == "hybrid_eho_aco"]["gap_pct"] < 1e-6).mean())
lr = {}
for topo in ["Cogentco", "Kdl", "syn500"]:
    s = summ[summ.topology == topo].copy()
    s["r"] = s.groupby(["k", "rho"])["F_mean"].rank()
    lr[topo] = s.groupby("algorithm")["r"].mean().sort_values().round(2).to_dict()
out["placement_large_ranks"] = lr
out["placement_runtime_s"] = summ.groupby("algorithm")["seconds"].mean().round(2).to_dict()
out["placement_runs_per_cell"] = int(summ["runs"].max())

# coupled, leakage inflation
out["coupled"] = pd.read_csv(O / "coupled" / "min_controllers.csv").to_dict("records")
out["inflation"] = pd.read_csv(O / "paper" / "tables" / "leakage_inflation.csv").round(4).to_dict("records")

# cross (3 seeds)
fs = pd.read_csv(O / "cross" / "fewshot_summary.csv")
piv = fs.pivot_table(index=["source", "target", "task", "model", "fraction"], columns="kind", values=["mean", "std"]).reset_index()
piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]
piv["ft_minus_scratch"] = piv["mean_finetuned"] - piv["mean_scratch"]
out["fewshot"] = piv.round(4).to_dict("records")
out["fewshot_not_better"] = int((piv["ft_minus_scratch"] <= 0).sum())
out["fewshot_cells"] = int(len(piv))
zs = pd.read_csv(O / "cross" / "zero_shot_summary.csv", header=[0, 1], index_col=[0, 1, 2, 3, 4])
zs.columns = ["_".join(c) for c in zs.columns]
zs = zs.reset_index()
out["zero_shot"] = zs.round(4).to_dict("records")
z = {(r["source"], r["target"], r["task"], r["model"], r["norm"]): r["macro_f1_mean"] for r in out["zero_shot"]}
keys = {k[:4] for k in z}
out["target_norm_better"] = sorted([list(k) for k in keys if z[k + ("target",)] > z[k + ("source",)]])
out["target_norm_cells"] = len(keys)
out["unseen"] = pd.read_csv(O / "cross" / "unseen_summary.csv").round(4).to_dict("records")
sa = j(O / "cross" / "shap_agreement.json")
out["shap_agreement"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("top10_x")} for k, v in sa.items()}

# latency (quick-pass file, source of truth)
lat = j(O / "latency" / "latency.json")
out["latency"] = {t: {m: {k: v[m].get(k) for k in ("throughput_b256", "throughput_b1", "size_kb", "params")}
                      for m in ("teacher", "shield", "shield_int8") if m in v}
                  for t, v in lat.items() if isinstance(v, dict) and "teacher" in v}

out["n_tables"] = len(list((O / "paper" / "tables").glob("*")))
out["n_figures"] = len(list((O / "paper" / "figures").glob("*")))
json.dump(out, open(Path(__file__).with_name("final_results.json"), "w"), indent=1, default=str)
print("ok")
