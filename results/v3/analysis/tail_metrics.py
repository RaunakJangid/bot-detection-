"""Tail and budget metrics for the false-alarm transfer (reporting style of Ha & Kim 2026, Appl. Sci. 16:2284).

Violation ratio rho = realised test benign FPR / target FPR, one value per run (model x seed). Per setting and model
family: mean rho, p90 and p95 of rho, StrictOK = share of runs with rho <= 1, B3 / B10 = share with rho <= 3 / 10.
Target 1% everywhere (IoT-23 also 0.1% and 5%). Rows "after recalibration": CIC settings with the clean 1,000-flow
time-ordered burn-in (recalib.json, median over draws); UNSW with 1,000-flow commissioning.
Sources: outputs_v3/recalib/recalib.json (E1), outputs_v3/iot23/scores_*.npz (E3), outputs_v3/unsw.log (E7).
Output: results/v3/tables/tail_metrics.csv."""

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

O = Path(r"D:\shield_run\outputs_v3")
OUT = Path(__file__).resolve().parents[1] / "tables" / "tail_metrics.csv"
rows = []


def summ(setting, stage, target, fam, ratios):
    r = np.asarray(ratios, float)
    rows.append({"setting": setting, "stage": stage, "target_pct": 100 * target, "family": fam, "runs": len(r),
                 "mean_ratio": r.mean(), "p90": np.quantile(r, 0.90), "p95": np.quantile(r, 0.95),
                 "strict_ok": (r <= 1).mean(), "b3_ok": (r <= 3).mean(), "b10_ok": (r <= 10).mean()})


rec = json.load(open(O / "recalib" / "recalib.json"))
for task, models in rec.items():
    by = defaultdict(lambda: defaultdict(list))
    for m, v in models.items():
        fam = v.get("family", "other")
        by["fixed"][fam].append(v["fixed"]["fpr"] / 0.01)
        if "clean_K1000_pi0.0" in v:
            by["recalibrated"][fam].append(v["clean_K1000_pi0.0"]["fpr_median"] / 0.01)
    for stage, d in by.items():
        for fam, rs in sorted(d.items()):
            summ(task, stage, 0.01, fam, rs)
        summ(task, stage, 0.01, "ALL", [x for rs in d.values() for x in rs])


def fam_iot(m):
    return ("gbdt" if m.startswith(("xgboost", "rf")) else "ensemble" if m.startswith("ensemble") else
            "large_neural" if m.startswith("large") else "small_mlp_kd" if "kd" in m else "small_mlp_scratch")


for design in ("same_family", "capture_shift", "random_control"):
    Z = np.load(O / "iot23" / f"scores_{design}.npz")
    for a in (0.001, 0.01, 0.05):
        d = defaultdict(list)
        for i, m in enumerate(Z["models"]):
            sv, st = Z["sv"][i], Z["st"][i]
            thr = np.quantile(sv[Z["yv"] == 0], 1 - a)
            d[fam_iot(str(m))].append(float((st[Z["yt"] == 0] > thr).mean()) / a)
        for fam, rs in sorted(d.items()):
            summ(f"iot23_{design}", "fixed", a, fam, rs)
        summ(f"iot23_{design}", "fixed", a, "ALL", [x for rs in d.values() for x in rs])

fx, cm = defaultdict(list), defaultdict(list)
for line in open(O / "unsw.log", encoding="utf8", errors="ignore"):
    mm = re.search(r"unsw \| (\w+_s\d): lab recall.*?\| (\{.*\})$", line.strip())
    if not mm:
        continue
    name, dd = mm.group(1), ast.literal_eval(mm.group(2))
    fam = "gbdt" if name.startswith("xgb") else "large_neural" if name.startswith("large") else \
          "small_mlp_kd" if name.startswith("small_kd") else "small_mlp_scratch"
    for a_, b_, *_ in dd.values():
        fx[fam].append(a_ / 1.0); cm[fam].append(b_ / 1.0)          # percentages / 1% target
for stage, d in (("fixed", fx), ("recalibrated", cm)):
    for fam, rs in sorted(d.items()):
        summ("unsw_realhome", stage, 0.01, fam, rs)
    summ("unsw_realhome", stage, 0.01, "ALL", [x for rs in d.values() for x in rs])

OUT.parent.mkdir(parents=True, exist_ok=True)
keys = list(rows[0])
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader()
    for r in rows:
        w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
print(f"{'setting':26s} {'stage':12s} {'tgt%':>5s} {'family':18s} {'runs':>4s} {'mean':>6s} {'p90':>6s} {'p95':>6s} {'ok':>5s} {'B3':>5s} {'B10':>5s}")
for r in rows:
    if r["family"] == "ALL" or r["setting"].startswith(("unsw",)):
        print(f"{r['setting']:26s} {r['stage']:12s} {r['target_pct']:5.1f} {r['family']:18s} {r['runs']:4d} {r['mean_ratio']:6.2f} "
              f"{r['p90']:6.2f} {r['p95']:6.2f} {r['strict_ok']:5.2f} {r['b3_ok']:5.2f} {r['b10_ok']:5.2f}")
