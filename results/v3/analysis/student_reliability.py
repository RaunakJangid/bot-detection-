"""Tiny-student operating-point reliability across every shift setting we have (P1, student angle).

Families: gbdt/trees, ensemble, large neural, small MLP distilled (the student size class), small MLP from scratch.
Per setting: median realised benign FPR at a 1% validation-set target (fixed threshold), and where available after
recalibration. Sources (no new training):
  CICIoMT x3 tasks and CICIoT 34 (control): outputs_v3/fpr_variance/fpr_variance.json (E2)
  IoT-23 per design: outputs_v3/iot23/scores_{design}.npz (E3; threshold from validation scores, pooled test)
  UNSW real home, 3 days: outputs_v3/unsw.log (E7; fixed and 1,000-flow commissioning)
Output: results/v3/tables/student_reliability.csv + printed table."""

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

O = Path(r"D:\shield_run\outputs_v3")
OUT = Path(__file__).resolve().parents[1] / "tables" / "student_reliability.csv"
FAMS = ["gbdt", "ensemble", "large_neural", "small_mlp_kd", "small_mlp_scratch"]
rows = []

f = json.load(open(O / "fpr_variance" / "fpr_variance.json"))
for task, v in f.items():
    med = v["fixed_1pct"]["family_median_fpr_pct"]
    rows.append({"setting": f"{task} (E2)", "metric": "FPR% @1% target, fixed", **{k: med.get(k) for k in FAMS},
                 "kd_vs_scratch_p": v["fixed_1pct"].get("kd_vs_scratch_p")})


def fam_iot(m):
    if m.startswith(("xgboost", "rf")):
        return "gbdt"
    if m.startswith("ensemble"):
        return "ensemble"
    if m.startswith("large_mlp"):
        return "large_neural"
    return "small_mlp_kd" if m.startswith("small_mlp_kd") else "small_mlp_scratch"


for design in ("same_family", "capture_shift", "random_control"):
    Z = np.load(O / "iot23" / f"scores_{design}.npz")
    fpr, auc = defaultdict(list), defaultdict(list)
    for i, m in enumerate(Z["models"]):
        sv, st, yv, yt = Z["sv"][i], Z["st"][i], Z["yv"], Z["yt"]
        t = np.quantile(sv[yv == 0], 0.99)
        fpr[fam_iot(str(m))].append(100 * float((st[yt == 0] > t).mean()))
        auc[fam_iot(str(m))].append(float(roc_auc_score(yt, st)))
    rows.append({"setting": f"iot23_{design} (E3)", "metric": "FPR% @1% target, fixed",
                 **{k: float(np.median(fpr[k])) if fpr[k] else None for k in FAMS}})
    rows.append({"setting": f"iot23_{design} (E3)", "metric": "test ROC-AUC (pooled)",
                 **{k: float(np.median(auc[k])) if auc[k] else None for k in FAMS}})

fx, cm = defaultdict(list), defaultdict(list)
for line in open(O / "unsw.log", encoding="utf8", errors="ignore"):
    m = re.search(r"unsw \| (\w+_s\d): lab recall.*?\| (\{.*\})$", line.strip())
    if not m:
        continue
    name, d = m.group(1), ast.literal_eval(m.group(2))
    fam = "gbdt" if name.startswith("xgb") else "large_neural" if name.startswith("large") else \
          "small_mlp_kd" if name.startswith("small_kd") else "small_mlp_scratch"
    for day, (a, b, *_rest) in d.items():
        fx[fam].append(a); cm[fam].append(b)
rows.append({"setting": "unsw_realhome (E7)", "metric": "FPR% @1% target, fixed", **{k: float(np.median(fx[k])) if fx[k] else None for k in FAMS}})
rows.append({"setting": "unsw_realhome (E7)", "metric": "FPR% after 1,000-flow commissioning", **{k: float(np.median(cm[k])) if cm[k] else None for k in FAMS}})

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, ["setting", "metric", *FAMS, "kd_vs_scratch_p"]); w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
print(f"{'setting':32s} {'metric':38s} " + " ".join(f"{k[:9]:>9s}" for k in FAMS))
for r in rows:
    print(f"{r['setting']:32s} {r['metric']:38s} " + " ".join(f"{fmt(r.get(k)):>9s}" for k in FAMS))
