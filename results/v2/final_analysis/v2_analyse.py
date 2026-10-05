"""v2 numbers for FINAL_REPORT_v2.md, with v1 values alongside. Read-only over D:/shield_run/outputs_v2 and outputs (v1)."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = Path(r"D:/shield_run/outputs_v2")
V1 = json.load(open(r"D:/shield_run/outputs/final_analysis/final_results.json"))
MAIN = {"ciciot": ["binary", "family", "class34"], "ciciomt": ["binary", "category", "attack"],
        "botiot": ["binary", "category"]}
VARIANTS = ["scratch_all", "kd_all", "kd_shap", "shield", "kd_random", "kd_mi", "kd_variance"]
out = {}


def j(p):
    return json.load(open(p, encoding="utf-8"))


def ms(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


def per_seed(ds, t, v):
    rows = []
    for p in sorted(glob.glob(str(O / "kd" / f"{ds}_{t}" / f"{v}_s[0-9]" / "metrics.json"))):
        rows.append(j(p))
    return rows


kd = {}
for ds, tasks in MAIN.items():
    for t in tasks:
        key = f"{ds}_{t}"
        row = {}
        for v in VARIANTS:
            ms_ = per_seed(ds, t, v)
            f = [m["test"]["macro_f1"] for m in ms_]
            q = [m["test_int8"]["macro_f1"] for m in ms_]
            agr = [m["agreement"]["teacher"] for m in ms_ if isinstance(m.get("agreement"), dict)]
            fid = [m["fidelity"]["spearman"] for m in ms_ if m.get("fidelity")]
            row[v] = {"f1": ms(f), "int8_drop": ms(np.array(f) - np.array(q)), "per_seed": f,
                      "agreement": ms(agr) if agr else None, "fidelity": ms(fid) if fid else None,
                      "k": ms_[0]["k"] if ms_ else None, "params": ms_[0]["params"] if ms_ else None}
        tm = [j(p)["test"]["macro_f1"] for p in sorted(glob.glob(str(O / "teacher" / f"{key}_resmlp_s*" / "metrics.json")))]
        th = [len(j(p)["history"]) for p in sorted(glob.glob(str(O / "teacher" / f"{key}_resmlp_s*" / "metrics.json")))]
        row["teacher"] = {"f1": ms(tm), "epochs_run": th}
        sh, ks = np.array(row["shield"]["per_seed"]), np.array(row["kd_shap"]["per_seed"])
        row["shield_minus_kd_shap"] = float((sh - ks).mean())
        row["kd_minus_scratch"] = float(row["kd_all"]["f1"]["mean"] - row["scratch_all"]["f1"]["mean"])
        row["shield_minus_scratch"] = float(row["shield"]["f1"]["mean"] - row["scratch_all"]["f1"]["mean"])
        row["teacher_minus_shield"] = float(row["teacher"]["f1"]["mean"] - row["shield"]["f1"]["mean"])
        v1 = V1["kd"][key]
        row["v1"] = {m: v1[m]["f1"]["mean"] for m in ["teacher", "scratch_all", "kd_all", "kd_shap", "shield"]}
        row["v1"]["shield_int8_drop"] = v1["shield"]["int8_drop"]["mean"]
        kd[key] = row
out["kd"] = kd

a = [x for r in kd.values() for x in r["shield"]["per_seed"]]
b = [x for r in kd.values() for x in r["kd_shap"]["per_seed"]]
out["shield_vs_kd_shap"] = {"n": len(a), "p": float(wilcoxon(a, b).pvalue), "mean_diff_points": float(100 * (np.mean(a) - np.mean(b))),
                            "shield_better": int(sum(x > y for x, y in zip(a, b)))}


def cd_groups(ranks, cd):
    items = sorted(ranks.items(), key=lambda kv: kv[1])
    best = items[0][1]
    cliques = []
    for i, (k, r) in enumerate(items):
        grp = [kk for kk, rr in items[i:] if rr - r < cd]
        if len(grp) > 1 and not any(set(grp) <= set(c) for c in cliques):
            cliques.append(grp)
    return {"order": [(k, round(r, 3)) for k, r in items], "tied_with_best": [k for k, r in items if r - best < cd], "cliques": cliques}


kr = j(O / "paper" / "tables" / "kd_ablation_ranks.json")
out["kd_cd"] = {"p": kr["friedman_p"], "cd": kr["nemenyi_cd"], **cd_groups(kr["average_rank"], kr["nemenyi_cd"])}
pf = j(O / "placement" / "friedman.json")
out["pl_cd"] = {"p": pf["friedman_p"], "cd": pf["nemenyi_cd"], "instances": pf["instances"], **cd_groups(pf["average_rank"], pf["nemenyi_cd"])}
out["v1_kd_cd"], out["v1_pl_cd"] = V1["kd_cd"], V1["placement_cd"]

summ = pd.read_csv(O / "placement" / "summary.csv")
gap = summ.dropna(subset=["gap_pct"])
out["pl_gap"] = gap.groupby("algorithm")["gap_pct"].agg(["mean", "median", "max"]).round(3).to_dict("index")
out["pl_zero_share"] = {a_: float((gap[gap.algorithm == a_]["gap_pct"] < 1e-6).mean()) for a_ in ["hybrid_eho_aco", "sa", "hybrid_no_ants"]}
lr = {}
for topo in ["Cogentco", "Kdl", "syn500"]:
    s = summ[summ.topology == topo].copy()
    s["r"] = s.groupby(["k", "rho"])["F_mean"].rank()
    lr[topo] = s.groupby("algorithm")["r"].mean().sort_values().round(2).to_dict()
out["pl_large"] = lr
out["pl_runtime"] = summ.groupby("algorithm")["seconds"].mean().round(2).to_dict()
w = pd.read_csv(O / "placement" / "wilcoxon.csv")
out["pl_wilcoxon"] = w.to_dict("records")
out["v1_pl_gap"] = V1["placement_gap"]

zs = pd.read_csv(O / "cross" / "zero_shot_summary.csv", header=[0, 1], index_col=[0, 1, 2, 3, 4])
zs.columns = ["_".join(c) for c in zs.columns]
out["zero_shot"] = zs.reset_index().round(4).to_dict("records")
fs = pd.read_csv(O / "cross" / "fewshot_summary.csv")
piv = fs.pivot_table(index=["source", "target", "task", "model", "fraction"], columns="kind", values="mean").reset_index()
piv["d"] = piv["finetuned"] - piv["scratch"]
out["fewshot_not_better"] = int((piv["d"] <= 0).sum())
out["fewshot_cells"] = int(len(piv))
out["unseen"] = pd.read_csv(O / "cross" / "unseen_summary.csv").round(4).to_dict("records")
out["shap_agreement"] = j(O / "cross" / "shap_agreement.json")

lat = j(O / "latency" / "latency.json")
out["latency"] = {t: {m: {k: v[m].get(k) for k in ("throughput_b256", "size_kb", "params", "n_features")}
                      for m in ("teacher", "shield", "shield_int8") if m in v}
                  for t, v in lat.items() if isinstance(v, dict) and "teacher" in v}
out["n_tables"] = len(list((O / "paper" / "tables").glob("*")))
out["n_figures"] = len(list((O / "paper" / "figures").glob("*")))
json.dump(out, open(Path(__file__).with_name("v2_results.json"), "w"), indent=1, default=str)
print("ok")
