"""P2 v4: claim-evidence map, generated from the frozen result files only (no hand-typed numbers).

For every claim of papers/novelty/P2_V4_CLAIMS_OUTLINE.md: the numbers the paper may use, the file and field they come
from, and the qualifier that must accompany them. Output: outputs_v5/p2/claim_map.json and
papers/novelty/P2_CLAIM_EVIDENCE_MAP.md (regenerate after any result changes)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

O = Path("D:/shield_run/outputs_v5/p2"); MD = Path("D:/shield_run/papers/novelty/P2_CLAIM_EVIDENCE_MAP.md")
J = lambda f: json.load(open(O / f))
L = lambda f: [json.loads(l) for l in open(O / f)]
r2 = lambda x, d=2: None if x is None else round(float(x), d)
rng = lambda v, d=2: f"{r2(min(v), d)} to {r2(max(v), d)}"


def c1():
    Q = J("quota_effect_seeds.json"); S = J("seed_robustness.json"); out = {}
    for k in ("ciciot_class34", "ciciomt_binary", "ciciomt_category", "ciciomt_attack"):
        P = Q[k]["pairs"]; s = Q[k]["summary"]
        rec = lambda pol: [100 * (p[f"f1_{pol}"] - p["f1_student"]) / (p["f1_expert"] - p["f1_student"]) for p in P
                           if p["f1_expert"] - p["f1_student"] > 0.5]
        out[k] = {"n_pairs": len(P), **{f"{m}_{pol}_median": r2(s[f"{m}_{pol}"]["median"]) for m in ("acc", "f1")
                                         for pol in ("student", "fixed", "quota", "expert")},
                  "quota_minus_fixed_f1": [r2(s["f1_quota_minus_fixed"]["median"]), s["f1_quota_vs_fixed_counts"]],
                  "quota_minus_fixed_acc": [r2(s["acc_quota_minus_fixed"]["median"]), s["acc_quota_vs_fixed_counts"]],
                  "expert_gain_recovered_pct_fixed_median": r2(np.median(rec("fixed")), 0) if rec("fixed") else None,
                  "expert_gain_recovered_pct_quota_median": r2(np.median(rec("quota")), 0) if rec("quota") else None,
                  "pairs_with_expert_gain_gt_0.5pt": len(rec("fixed")),
                  "expert_vs_student_seed_pairs": S[k]["favour_counts_all"], "expert_minus_student_f1_median": r2(S[k]["f1_diff_median"])}
    return out


def c2():
    T = J("iot23_truemix.json"); out = {}
    for k, v in T.items():
        w = v["weightings"]
        out[k.split("|")[1][-4:]] = {"coverage_pct": r2(100 * v["coverage"], 1),
                                     "escalated_share_as_tested_pct": rng([100 * x for x in w["distinct"]["escalated_share_student"]], 1),
                                     "escalated_share_traffic_weighted_pct": rng([100 * x for x in w["volume"]["escalated_share_student"]], 2)}
    return out


def c3():
    A = J("escalation_attack.json")["rows"]; PC = J("policy_compare.json"); QR = J("quota_robust.json")
    TE = J("traffic_escalation.json")["aggregation"]; F = J("quota_competition.json")["failover"]
    by = defaultdict(list)
    for r in A:
        by["quota_block_max"].append(r["block_max_quota"]); by["quota_ratio_worst"].append(r["ratio_worst_quota"])
        if abs(r["rho"] - 3.2) < 1e-9:
            by["nocap_block_max_rho3.2"].append(r["block_max_nocap"]); by["online_block_max_rho3.2"].append(r["block_max_online"])
            by["nocap_ratio_rho3.2"].append(r["ratio_worst_nocap"]); by["online_ratio_rho3.2"].append(r["ratio_worst_online"])
    pol = defaultdict(list)
    for r in PC["streams"] + PC["flood"]:
        pol[r["policy"]].append(r["block_max"])
    adm = [r["adm_failed_merged"] / max(r["adm_failed_alone"], 1) for r in F]
    unfl = [r for r in F if not any(f in r["surviving"] + r["failed"] for f in ("35-1", "49-1"))]
    lost = [r["corrections_alone"] - r["corrections_merged"] for r in F]
    return {"X12_flooding_1000flow_block_max_quota": rng(by["quota_block_max"], 3),
            "X12_capacity_ratio_worst_block_quota": rng(by["quota_ratio_worst"]),
            "X12_rho3.2_block_max_nocap_vs_online": [rng(by["nocap_block_max_rho3.2"], 3), rng(by["online_block_max_rho3.2"], 3)],
            "X12_rho3.2_capacity_ratio_nocap_vs_online": [rng(by["nocap_ratio_rho3.2"]), rng(by["online_ratio_rho3.2"])],
            "T4_block_max_by_policy_max_over_runs": {k: r2(max(v), 3) for k, v in pol.items()},
            "OptionA_controllers": {n: {"quota_p": v["nominal_p"], "exact": v["exact"], "cap_only_robust_p": v["cap_robust_p"],
                                        "heldout_overload": v["heldout_overload_U0"]} for n, v in QR.items()},
            "s103_exceed_by_20pct": {f"{w}s": {k: r2(100 * TE[w][k]["share_exceeding_prediction_by_20pct"], 1)
                                               for k in ("nominal", "window_global", "nominal_quota", "window_global_quota")} for w in TE},
            "s104_failover_admission_factor": [r2(np.median(adm)), rng(adm)],
            "s104_corrections_lost_all": {"median": r2(np.median(lost), 1), "max": max(lost), "cases_with_loss": sum(x > 0 for x in lost), "n": len(lost)},
            "s104_unflagged_cases": len(unfl)}


def c4():
    ab, abs_ = J("ablation_stats.json"), J("ablation_shift_stats.json"); SF = J("shift_feasibility.json"); RG = J("runtime_gap.json")
    C = L("modelb_close.jsonl"); D = L("modelb_draws.jsonl")
    pick = lambda st, b, sub=None: {a: (t["full_better"], t["full_worse"], r2(t["p_holm"], 3))
                                     for a, t in (st[b][sub] if sub else st[b])["full_vs"].items()}
    gap = Counter(r["ub"] - r["lb"] for r in C)
    # grid ratios only: all 120 rows have them; measured-scenario keys exist only in the 8 rows of scripts/79b (the 112
    # rows of scripts/79 predate capacity_v4.json) and are mapped onto the monotone grid instead of being quoted alone
    grid = sorted({k for d in D for k in d["scenarios"] if k.startswith("r")}, reverse=True)
    pf = {k: [r2(100 * np.mean([d["scenarios"][k]["planned_feasible"] for d in D]), 0), len(D)] for k in grid}
    return {"ILP_runtime_median_s": r2(RG["ilp"]["median_s"], 1), "n_proven": RG["ilp"]["n_proven"],
            "EHOACO_runtime_opt_hit": {k: (r2(v["median_s"]), r2(100 * v["opt_hit"], 1)) for k, v in RG["small"].items()},
            "FFCCP_ablation_1000": pick(ab, "1000", "all"), "FFCCP_ablation_20000": pick(ab, "full", "all"),
            "shift_ablation_1000": pick(abs_, "1000"), "shift_ablation_20000": pick(abs_, "full"),
            "X13_feasible_yield": {a: r2(v["feasible_yield"], 3) for a, v in SF["per_alg"].items()},
            "X13_vs": {"quality": {a: (v["eho_better"], v["eho_worse"], r2(v["p_holm"], 4)) for a, v in SF["conditional_quality_vs"].items()},
                       "yield": {a: (v["eho_higher"], v["eho_lower"], r2(v["p_holm"], 4)) for a, v in SF["feasible_yield_vs"].items()}},
            "X5_rows": len(D), "X5_planned_set_feasible_pct_by_capacity_ratio": pf,
            "X5_closure": {"n": len(C), "exact_pct": r2(100 * np.mean([r["exact"] for r in C]), 0), "gap_counts": dict(sorted(gap.items()))}}


def c5():
    out = {}
    for lab, files, key in (("IoT-23", [f"capacity_curve_iot23_quiet_r{i}.json" for i in (1, 2, 3)], "iot23"),
                            ("CIC 6-class", [f"capacity_curve_quiet_r{i}.json" for i in (1, 4, 5, 6)], "ciciomt_category"),
                            ("CIC 19-class", [f"capacity_curve_quiet_r{i}.json" for i in (1, 4, 5, 6)], "ciciomt_attack")):
        runs = [J(f) if key == "iot23" else J(f)[key] for f in files]
        row = {"runs": len(runs)}
        for e in ("0.0", "0.1", "0.2"):
            for b in ("b1", "b256"):
                v = [r["curve"][e][f"{b}_throughput"] for r in runs]
                row[f"{b}_e{e}"] = f"median {r2(np.median(v), 0)} ({rng(v, 0)})"
        ratio = [r["separate"]["1"]["mu_s"] / r["separate"]["1"]["mu_t"] for r in runs]
        row["mu_s_over_mu_t_b1"] = rng(ratio, 1)
        out[lab] = row
    return out


def x7():
    R = L("x7/runs.jsonl"); g = defaultdict(list)
    for r in R:
        g[(r["setting"], r["sizing"], r["mode"])].append(r)
    return {f"{s}|{z}|{m}": {"reps": len(v), "answered_pct": rng([100 * x["answered_frac"] for x in v], 1),
                             "escalated_pct": rng([100 * x["escalated_frac"] for x in v], 1),
                             "lat_esc_p99_ms": rng([x["lat_esc_p99_ms"] for x in v], 0), "macro_f1": rng([x["macro_f1"] for x in v], 2)}
            for (s, z, m), v in sorted(g.items())}


def main():
    M = {"C1_quality": c1(), "C2_iot23_demand": c2(), "C3_quota_bound": c3(), "C4_placement": c4(), "C5_cost": c5(), "X7_mininet": x7()}
    json.dump(M, open(O / "claim_map.json", "w"), indent=1, default=str)
    Q = {"C1_quality": "Accuracy and macro-F1 separately; crossed seed pairs = directional consistency; CICIoT2023 = no-shift control; recovery share only over pairs where the expert gains > 0.5 macro-F1; quota quality effect is measured, not guaranteed.",
         "C2_iot23_demand": "As tested = de-duplicated test vectors; traffic-weighted only with coverage; test-label-free but model-mediated; IoT-23 is not quality evidence (label note).",
         "C3_quota_bound": "The quota bounds the escalated FRACTION per 300-flow window, not the absolute rate. scripts/103: nominal_quota = 0% holds by construction (cap equals the prediction) and is NOT evidence; 103 uses sampler-only weights and synthetic co-occurrence of separately recorded captures; give each number with its predictor (nominal 10% or window-global) and window (60 s / 300 s). Failover: label-free admission factor only; correction counts only for unflagged captures.",
         "C4_placement": "No hybrid-synergy claim (pre-specified 1,000 / 20,000). SA alone best at 20,000 on FFCCP. X13 = under-shift controller-placement benchmark; 27% scenario = stress scenario from as-tested IoT-23 1-1. X5 p_min reported as [LB, UB] from the closure, never raw time-limited values; new SURFNET K = 3 rows overlapped a GROMACS job (bounds valid, less tight).",
         "C5_cost": "Quiet runs only (IoT-23 r1-r3, CIC r1, r4-r6); median and range; IoT-23 r2 slow, unexplained; cost at prescribed escalation shares; m and eta cancel in U(e)/U0.",
         "X7_mininet": "'cap' = online 90th-percentile cap (the refinement), NOT the quota; quota mode pending (queued until GROMACS ends)."}
    lines = ["# P2 v4 claim-evidence map (generated by scripts/117_p2v4_claim_map.py; do not edit by hand)", "",
             "Every number below is read from the frozen files in outputs_v5/p2 (see runs/evidence_freeze/manifest.txt).", ""]
    for k, v in M.items():
        lines += [f"## {k}", "", f"**Mandatory qualifier:** {Q[k]}", "", "```", json.dumps(v, indent=1, default=str), "```", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(M, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
