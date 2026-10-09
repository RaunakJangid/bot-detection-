"""Build results/v3: every table of FINAL_REPORT_v3.md is generated here from the experiment outputs (no
hand-copied numbers). Tables -> results/v3/tables/*.csv; the report template results/v3/REPORT_TEMPLATE.md
has {{name}} placeholders that are replaced by the markdown of table `name` -> results/v3/FINAL_REPORT_v3.md.
Also copies the small JSON summaries the tables come from into results/v3/data/."""

import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from shield.utils.config import PROJECT_ROOT, Paths

P = Paths()
V2, V3, FULL = P.outputs, P.v3, P.outputs.with_name("outputs_full")
OUT = PROJECT_ROOT / "results" / "v3"
ANA = OUT / "analysis"
TABLES: dict[str, pd.DataFrame] = {}
TASKS = [("ciciot", "binary"), ("ciciot", "family"), ("ciciot", "class34"), ("ciciomt", "binary"),
         ("ciciomt", "category"), ("ciciomt", "attack"), ("botiot", "binary"), ("botiot", "category")]
NAME = {"ciciot_binary": "CICIoT binary", "ciciot_family": "CICIoT 8-class", "ciciot_class34": "CICIoT 34-class",
        "ciciomt_binary": "CICIoMT binary", "ciciomt_category": "CICIoMT 6-class", "ciciomt_attack": "CICIoMT 19-class",
        "botiot_binary": "Bot-IoT binary", "botiot_category": "Bot-IoT 5-class"}


def lj(p):
    return json.load(open(p, encoding="utf-8"))


def add(name, df, index=False):
    TABLES[name] = (df, index)


def to_md(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:g}"
        return str(v).replace("|", "/")
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def pct(x, d=2):
    return round(100 * float(x), d)


# ---------------------------------------------------------------- step 3: identifiability
kc = lj(ANA / "knn_ceiling.json")
rows = []
for key, v in kc.items():
    ds, task = key.split("/")
    m = {"teacher": lj(V2 / "teacher" / f"{ds}_{task}_resmlp_s0" / "metrics.json")["test"]["accuracy"],
         "xgb": lj(V2 / "xgboost" / f"{ds}_{task}_s0" / "metrics.json")["test"]["accuracy"]}
    rows.append({"task": NAME[f"{ds}_{task}"], "1-NN error (val) %": pct(v["err_1nn"]),
                 "Bayes error bound % (Cover-Hart)": f"{pct(v['bayes_lower'])} - {pct(v['err_1nn'])}",
                 "XGBoost test error %": pct(1 - m["xgb"]), "Teacher s0 test error %": pct(1 - m["teacher"])})
add("ceiling", pd.DataFrame(rows))

# ---------------------------------------------------------------- step 2: decision rules
ph = pd.read_csv(V3 / "posthoc" / "summary.csv")
ph["tk"] = (ph.dataset + "_" + ph.task).map(NAME)
main = ph[ph.dataset.isin(["ciciot", "ciciomt", "botiot"])]
w = []
for model, g in main.groupby("model"):
    for rule in ("la", "bias"):
        d = (g[f"test_{rule}"] - g["test_raw"]).to_numpy()
        w.append({"model": model, "rule": {"la": "logit adjustment", "bias": "LA + per-class bias"}[rule],
                  "pairs": len(d), "mean gain (pts)": round(100 * d.mean(), 2), "better": int((d > 1e-12).sum()),
                  "worse": int((d < -1e-12).sum()),
                  "p (Wilcoxon, two-sided)": float(f"{wilcoxon(d, zero_method='zsplit').pvalue:.2g}") if np.any(d) else 1.0})
add("posthoc_tests", pd.DataFrame(w))
pt = (main[main.model.isin(["teacher", "xgboost", "kd_all", "shield"])]
      .groupby(["tk", "model"])[["test_raw", "test_bias"]].mean().mul(100).round(2))
pt = pt.unstack("model")
pt.columns = [f"{m} {'raw' if c == 'test_raw' else '+rule'}" for c, m in pt.columns]
add("posthoc_tasks", pt.reindex([NAME[f"{a}_{b}"] for a, b in TASKS]), index=True)

# ---------------------------------------------------------------- step 1: ensemble teacher + its students
rows = []
for ds, task in TASKS:
    m = lj(V3 / "teacher_ens" / f"{ds}_{task}" / "metrics.json")
    t2 = np.mean([lj(V2 / "teacher" / f"{ds}_{task}_resmlp_s{s}" / "metrics.json")["test"]["macro_f1"] for s in range(5)])
    rows.append({"task": NAME[f"{ds}_{task}"], "v2 teacher (5-seed mean)": pct(t2),
                 "XGBoost": pct(m["test_members"]["xgboost"]["macro_f1"]),
                 "deep ensemble (5 ResMLP)": pct(m["test_members"]["deep_ensemble"]["macro_f1"]),
                 "w_ResMLP (val)": m["weight_resmlp"], "ensemble": pct(m["test"]["macro_f1"]),
                 "ensemble + rule": pct(m["test_bias"]["macro_f1"])})
add("ensemble_teacher", pd.DataFrame(rows))
st = []
for new, old in (("v3-kd_all_ens", "kd_all"), ("v3-kd_shap_ens", "kd_shap"), ("v3-kd_shap_ens", "shield")):
    for col in ("test_raw", "test_bias"):
        a = main[main.model == new].set_index(["tk", "seed"])[col]
        b = main[main.model == old].set_index(["tk", "seed"])[col]
        d = (a - b).dropna()
        st.append({"comparison": f"{new[3:]} vs {old}", "decision": "raw" if col == "test_raw" else "+rule",
                   "pairs": len(d), "mean diff (pts)": round(100 * d.mean(), 2), "better": int((d > 0).sum()),
                   "p": float(f"{wilcoxon(d).pvalue:.2g}")})
add("ens_students", pd.DataFrame(st))

# ---------------------------------------------------------------- step 4: int8
i8 = pd.read_csv(V3 / "int8" / "summary.csv")
i8["task"] = (i8.dataset + "_" + i8.task).map(NAME)
t = i8.groupby("task")[["drop_pts_v2", "drop_pts", "drop_pts_w8a16"]].mean().round(2)
t.columns = ["v2 dynamic int8 drop (pts)", "v3 W8A8 drop (pts)", "v3 W8A16 drop (pts)"]
sh = i8[i8.variant == "shield"].groupby("task")[["fp32_bytes", "int8_bytes"]].first()
t = t.join(sh.rename(columns={"fp32_bytes": "SHIELD fp32 bytes", "int8_bytes": "SHIELD int8 bytes"}))
t.loc["mean (all 160 models)"] = [round(i8.drop_pts_v2.mean(), 2), round(i8.drop_pts.mean(), 2),
                                  round(i8.drop_pts_w8a16.mean(), 2), np.nan, np.nan]
add("int8", t.reindex([NAME[f"{a}_{b}"] for a, b in TASKS] + ["mean (all 160 models)"]), index=True)

# ---------------------------------------------------------------- step 5: cascade
cs = pd.read_csv(V3 / "cascade" / "rules" / "summary.csv")
cs["tk"] = (cs.dataset + "_" + cs.task).map(lambda k: NAME.get(k, k.replace("ciciot_full_", "CICIoT(full) ")))
c = cs[cs.dataset.isin(["ciciot", "ciciomt", "botiot"]) & (cs.student == "shield")]
rows = []
for (tk, ex), g in c.groupby(["tk", "expert"]):
    r = {"task": tk, "expert": ex, "student alone": pct(g[g.budget == 0].test_macro_f1.mean())}
    for b in (0.05, 0.1, 0.2):
        r[f"{int(100 * b)}% budget"] = pct(g[g.budget == b].test_macro_f1.mean())
    r["expert alone"] = pct(g.expert_test_f1.mean())
    m = g[g.matched]
    r["matched: F1"], r["matched: test escalated %"] = pct(m.test_macro_f1.mean()), pct(m.test_escalated.mean(), 1)
    rows.append(r)
add("cascade", pd.DataFrame(rows))

# ---------------------------------------------------------------- step 6: coupling
mk = pd.read_csv(V3 / "coupled" / "min_controllers.csv")
add("coupling", mk)
acc = lj(V3 / "coupled" / "coupled_meta.json")["accuracy"]
add("coupling_acc", pd.DataFrame([{"model": k, "test macro-F1 (CICIoT 34)": pct(v["test_macro_f1"]),
                                   "test escalated %": pct(v["test_escalated"], 1)} for k, v in acc.items()]))

# ---------------------------------------------------------------- explanation fidelity
fw = lj(V3 / "fidelity" / "wilcoxon.json")
fd = pd.read_csv(V3 / "fidelity" / "summary.csv")
fd["task"] = (fd.dataset + "_" + fd.task).map(NAME)
ft = fd.pivot_table(index="task", columns="variant", values="spearman").round(3)
ft.loc["mean over tasks"] = ft.mean().round(3)
add("fidelity", ft, index=True)
FID = fw

# ---------------------------------------------------------------- step 8: unseen attacks
un = pd.read_csv(V3 / "unseen" / "summary.csv")
u = un[un.fpr_target == 0.01].groupby(["dataset", "unseen", "detector"])[["detect_unseen", "test_fpr"]].mean().mul(100).round(1)
u = u.unstack("detector")
u.columns = [f"{d} {'detect %' if m == 'detect_unseen' else 'test FPR %'}" for m, d in u.columns]
add("unseen", u, index=True)

# ---------------------------------------------------------------- placement
pl = V3 / "placement"
ALG = ["hybrid_eho_aco", "hybrid_linear", "hybrid_fuzzy", "hybrid_no_ants", "hybrid_no_ls", "hybrid_swap_ls", "eho",
       "aco", "ga", "pso", "sa", "gwo", "gwo_sa", "foa", "woa", "hho", "de", "random"]
s = pd.read_csv(pl / "summary.csv")
gp = s.dropna(subset=["gap_pct"])
small = gp[[math.comb(int(n), int(k)) <= 2_000_000 for n, k in zip(gp.n, gp.k)]]
medium = gp[[math.comb(int(n), int(k)) > 2_000_000 for n, k in zip(gp.n, gp.k)]]
v2g = pd.read_csv(V2 / "placement" / "summary.csv").dropna(subset=["gap_pct"]).groupby("algorithm").gap_pct.mean()
runs = [json.loads(l) for l in open(pl / "runs.jsonl")]
rd = pd.DataFrame([{k: r[k] for k in ("topology", "k", "rho", "algorithm", "seed", "F", "n_evals")} | {"history": r.get("history")}
                   for r in runs if r["algorithm"] in ALG and not r.get("skipped")])
blk = rd.groupby(["topology", "algorithm"]).F.mean().unstack()[ALG]
inst_rank = s[s.algorithm.isin(ALG)].pivot_table(index=["topology", "k", "rho"], columns="algorithm", values="F_mean")[ALG].rank(axis=1).mean()
hist = rd[rd.history.map(lambda h: isinstance(h, list) and len(h) > 0)]
t99 = hist.groupby("algorithm").history.apply(
    lambda c: np.mean([np.argmax(np.asarray(v) <= v[-1] * 1.001) / (len(v) - 1) for v in c]))
rows = []
for a in ALG:
    rows.append({"algorithm": a, "gap % (22 small, v2 set)": round(small[small.algorithm == a].gap_pct.mean(), 3),
                 "gap % (all 37 exact)": round(gp[gp.algorithm == a].gap_pct.mean(), 3),
                 "gap % (15 medium/large exact)": round(medium[medium.algorithm == a].gap_pct.mean(), 3),
                 "worst gap %": round(gp[gp.algorithm == a].gap_pct.max(), 2),
                 "rank (56 instances)": round(inst_rank[a], 2), "rank (8 topologies)": round(blk.rank(axis=1).mean()[a], 2),
                 "budget fraction to final": round(t99.get(a, np.nan), 3),
                 "v2 gap % (22 small)": round(v2g.get(a, np.nan), 3)})
add("placement", pd.DataFrame(rows).sort_values("rank (8 topologies)"))
PL = {"friedman_p_topology": friedmanchisquare(*[blk[a] for a in ALG]).pvalue, "n_exact": gp.groupby(["topology", "k", "rho"]).ngroups,
      "evals": sorted(rd.n_evals.unique().tolist())}
inst = rd.groupby(["topology", "k", "rho", "algorithm"]).F.mean().unstack()
rows = []
for a, b in (("hybrid_eho_aco", "gwo_sa"), ("hybrid_linear", "gwo_sa"), ("hybrid_linear", "hybrid_eho_aco"),
             ("hybrid_fuzzy", "gwo_sa"), ("hybrid_fuzzy", "hybrid_eho_aco"), ("hybrid_eho_aco", "hybrid_no_ants")):
    x, tt = inst[a] - inst[b], blk[a] - blk[b]
    rows.append({"A": a, "B": b, "instances A better": int((x < -1e-9).sum()), "instances A worse": int((x > 1e-9).sum()),
                 "p (56 instances)": float(f"{wilcoxon(x, zero_method='zsplit').pvalue:.2g}"),
                 "topologies A better (of 8)": int((tt < 0).sum()), "p (8 topologies)": float(f"{wilcoxon(tt).pvalue:.2g}")})
add("placement_pairs", pd.DataFrame(rows))
wx = pd.read_csv(pl / "wilcoxon.csv")   # hybrid_eho_aco vs each method, per instance over 30 seeds, Holm
wx = wx.assign(better=wx.p_holm < 0.05, worse=wx.p_worse_holm < 0.05).groupby("vs")[["better", "worse"]].sum()
add("placement_holm", wx.loc[[a for a in ALG if a in wx.index]].rename(
    columns={"better": "hybrid_eho_aco significantly better (instances of 56)",
             "worse": "hybrid_eho_aco significantly worse"}), index=True)
tp = lj(pl / "tuned_params.json")
add("placement_tuning", pd.DataFrame([{"algorithm": a, "tuning score": round(tp[a]["score"], 4),
                                       "default score": round(tp[a]["default_score"], 4) if tp[a].get("default_score") else None}
                                      for a in ALG if a in tp and "inherited_from" not in tp[a]]).sort_values("tuning score"))

# ---------------------------------------------------------------- surge
sg = pd.read_csv(V3 / "surge" / "runs.csv")
t = sg[sg.M > 0].groupby(["M", "strategy"])[["benign_sla", "benign_drop", "drop_frac", "attack_exposure", "util_max",
                                             "k_total", "time_to_mitigate_s"]].mean()
for c_ in ("benign_sla", "benign_drop", "drop_frac", "attack_exposure"):
    t[c_] = (100 * t[c_]).round(2)
t[["util_max", "k_total", "time_to_mitigate_s"]] = t[["util_max", "k_total", "time_to_mitigate_s"]].round(2)
t.columns = ["benign within SLA %", "benign flows dropped %", "all flows dropped %", "attack exposure %",
             "max utilisation", "controllers", "time to mitigate s"]
add("surge", t, index=True)
p = sg[sg.M > 0].pivot_table(index=["topology", "M", "trial"], columns="strategy", values="benign_drop")
rows = []
for o in ("S1_replace_k+1", "S1_replace_k", "S0_static"):
    x = p["S2_isolate_k+1"] - p[o]
    rows.append({"isolation (S2) vs": o, "mean benign-drop diff (pts)": round(100 * x.mean(), 2),
                 "S2 lower in": f"{int((x < -1e-9).sum())}/{len(x)}", "p": float(f"{wilcoxon(x, zero_method='zsplit').pvalue:.2g}")})
add("surge_tests", pd.DataFrame(rows))
DET = lj(V3 / "surge" / "detection.json")

# ---------------------------------------------------------------- detector_v4 + full data
dv = pd.read_csv(V3 / "detector_v4" / "summary.csv") if (V3 / "detector_v4" / "summary.csv").exists() else None
dvj = [lj(f) for f in sorted((V3 / "detector_v4").glob("*.json"))]
rows = []
for j in dvj:
    r0 = j["rows"][0]
    key = f"{r0['dataset']}_{r0['task']}"
    rec = {"task": NAME.get(key, key.replace("ciciot_full_", "CICIoT(full) "))}
    for r in j["rows"]:
        rec[r["model"]] = pct(r["test_macro_f1"])
    rows.append(rec)
add("detector_v4", pd.DataFrame(rows))
rows = []
for task in ("binary", "family", "class34"):
    tsub = np.mean([lj(V2 / "teacher" / f"ciciot_{task}_resmlp_s{s}" / "metrics.json")["test"]["macro_f1"] for s in range(5)])
    tful = np.mean([lj(FULL / "teacher" / f"ciciot_full_{task}_resmlp_s{s}" / "metrics.json")["test"]["macro_f1"] for s in range(5)])
    xs = lj(V2 / "xgboost" / f"ciciot_{task}_s0" / "metrics.json")["test"]["macro_f1"]
    xf = lj(FULL / "xgboost" / f"ciciot_full_{task}_s0" / "metrics.json")["test"]["macro_f1"]
    es = [r for r in lj(V3 / "detector_v4" / f"ciciot_{task}.json")["rows"] if r["model"] in ("ens4", "ens4+rule")]
    ef = [r for r in lj(V3 / "detector_v4" / f"ciciot_full_{task}.json")["rows"] if r["model"] in ("ens4", "ens4+rule")]
    shs = ph[(ph.dataset == "ciciot") & (ph.task == task) & (ph.model == "shield")].test_bias.mean()
    shf = ph[(ph.dataset == "ciciot_full") & (ph.task == task) & (ph.model == "shield")].test_bias.mean()
    cf = cs[(cs.dataset == "ciciot_full") & (cs.task == task) & (cs.student == "shield") & (cs.expert == "ens4")]
    rows.append({"task": NAME[f"ciciot_{task}"], "teacher subset": pct(tsub), "teacher full": pct(tful),
                 "XGBoost subset": pct(xs), "XGBoost full": pct(xf),
                 "ens4 subset": pct(es[0]["test_macro_f1"]), "ens4 full": pct(ef[0]["test_macro_f1"]),
                 "ens4+rule subset": pct(es[1]["test_macro_f1"]), "ens4+rule full": pct(ef[1]["test_macro_f1"]),
                 "SHIELD+rule subset": pct(shs), "SHIELD+rule full": pct(shf),
                 "cascade 10% full": pct(cf[cf.budget == 0.1].test_macro_f1.mean()),
                 "cascade 20% full": pct(cf[cf.budget == 0.2].test_macro_f1.mean())})
add("full", pd.DataFrame(rows))
FM = lj(P.processed("ciciot_full") / "meta.json")["dedup"]
lc = lj(ANA / "learning_curve.json")
add("learning_curve", pd.DataFrame([{"train rows": r["n"], "fraction": r["frac"], "macro-F1 (val)": pct(r["macro_f1"]),
                                     "8 rare classes": pct(r["rare_f1"]), "other 26": pct(r["common_f1"])} for r in lc["points"]]))

add("leaderboards", pd.read_csv(OUT / "tables" / "leaderboards.csv"))   # hand-curated sources; see the CSV

# ---------------------------------------------------------------- outputs
(OUT / "tables").mkdir(parents=True, exist_ok=True)
md = {}
for name, (df, idx) in TABLES.items():
    df.to_csv(OUT / "tables" / f"{name}.csv", index=idx)
    md[name] = to_md(df.reset_index() if idx else df)
facts = {"fidelity": FID, "placement": PL, "surge_detection": DET, "full_dedup": FM,
         "learning_curve_fit": {k: v for k, v in lc.items() if k != "points"}}
(OUT / "data").mkdir(exist_ok=True)
json.dump(facts, open(OUT / "data" / "facts.json", "w"), indent=1, default=float)
tpl = OUT / "REPORT_TEMPLATE.md"
if tpl.exists():
    text = tpl.read_text(encoding="utf-8")
    missing = [m for m in re.findall(r"\{\{(\w+)\}\}", text) if m not in md]
    if missing:
        sys.exit(f"unknown table placeholders: {missing}")
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: md[m.group(1)], text)
    (OUT / "FINAL_REPORT_v3.md").write_text(text, encoding="utf-8")
print("tables:", ", ".join(TABLES))
print(json.dumps(facts, indent=1, default=float)[:3000])
