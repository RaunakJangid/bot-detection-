"""P2 v4: summary of X4 (Model A rerun) and X9 (headroom) from outputs_v5/p2/modelA_v4.jsonl.
Output: outputs_v5/p2/modelA_summary.json and a printed table.
Per topology and rho, over the 5 demand draws: k_min at the plan (r = 1) and at each scenario ratio (monotone envelope,
median and range), extra controllers over the plan, whether the planned placement stays feasible at the scenario, the
number of raw k_min values that violate monotonicity (search noise remaining), and for X9 the cost and survival of
planning with headroom (plan at r_plan, evaluate at the measured scenarios)."""
import json
from collections import defaultdict

import numpy as np

rows = [json.loads(l) for l in open("D:/shield_run/outputs_v5/p2/modelA_v4.jsonl")]
R = rows[0]["ratios"]
SCEN = {"b256_e16_ciciomt6": "16% CICIoMT 6-class", "b256_e17_ciciomt19": "17% CICIoMT 19-class",
        "b256_e27_iot23_1-1": "27% IoT-23 1-1", "b256_e13_online_cap_ciciomt6": "online cap (13%) CIC",
        "b256_e13_online_cap_iot23": "online cap (13%) IoT-23", "b1_e16_ciciomt6": "16% (batch 1)",
        "b1_e17_ciciomt19": "17% (batch 1)", "b1_e27_iot23_1-1": "27% (batch 1)"}
PLANS = ["plan", "b256_e13_online_cap_ciciomt6", "b256_e16_ciciomt6", "b256_e27_iot23_1-1"]
out = {"ratios": R, "groups": {}}
viol_total = 0
for r in rows:                                   # raw monotonicity violations: k_raw rising with capacity
    names = list(R)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):       # names sorted by decreasing ratio
            a, b = r["k_raw"][names[i]], r["k_raw"][names[j]]
            if a is not None and b is not None and a > b:
                viol_total += 1
out["raw_monotone_violations"] = viol_total
G = defaultdict(list)
for r in rows:
    G[(r["topology"], r["rho"])].append(r)
for (t, rho), rs in sorted(G.items()):
    g = {"n_draws": len(rs), "k_plan": [r["k_env"]["plan"] for r in rs], "scen": {}, "headroom": {}}
    for s in SCEN:
        ks = [r["k_env"][s] for r in rs]
        extra = [k - r["k_env"]["plan"] if k is not None else None for k, r in zip(ks, rs)]
        feas = [r["cross"]["plan"][s]["feasible"] for r in rs if "plan" in r["cross"]]
        g["scen"][s] = {"k": ks, "extra": extra, "plan_feasible": sum(feas), "n": len(feas)}
    for p in PLANS:                              # X9: plan at r_plan, evaluated at the measured b256 scenarios
        cost = [r["k_env"][p] - r["k_env"]["plan"] if r["k_env"][p] is not None else None for r in rs]
        surv = {s: sum(r["cross"][p][s]["feasible"] for r in rs if p in r["cross"]) for s in
                ("b256_e16_ciciomt6", "b256_e17_ciciomt19", "b256_e27_iot23_1-1", "b256_e13_online_cap_ciciomt6")}
        g["headroom"][p] = {"extra_controllers": cost, "survives": surv}
    out["groups"][f"{t}|{rho}"] = g
json.dump(out, open("D:/shield_run/outputs_v5/p2/modelA_summary.json", "w"), indent=1)
print("raw monotonicity violations over all draws:", viol_total)
for key, g in out["groups"].items():
    line = f"{key:16s} plan k {g['k_plan']}"
    for s in ("b256_e13_online_cap_ciciomt6", "b256_e16_ciciomt6", "b256_e17_ciciomt19", "b256_e27_iot23_1-1", "b1_e27_iot23_1-1"):
        v = g["scen"][s]; line += f" | {s.split('_', 1)[1][:14]} extra {v['extra']} planOK {v['plan_feasible']}/{v['n']}"
    print(line)
print("\nX9 headroom (extra controllers at plan time; survives at 16%/17%/27%/cap13 out of 5 draws):")
for key, g in out["groups"].items():
    print(key, {p.split('_', 1)[-1][:12]: (g["headroom"][p]["extra_controllers"], list(g["headroom"][p]["survives"].values())) for p in PLANS})
