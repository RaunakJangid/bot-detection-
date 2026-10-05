import json
r = json.load(open(r"D:\shield_run\outputs_v2\final_analysis\v2_results.json"))
V = ["teacher", "scratch_all", "kd_all", "kd_shap", "shield", "kd_random", "kd_mi", "kd_variance"]
print("task".ljust(17) + "".join(v[:10].rjust(15) for v in V) + "   k  params")
for t, row in r["kd"].items():
    cells = []
    for v in V:
        x = row[v]["f1"]
        cells.append(("%.2f+-%.2f" % (100 * x["mean"], 100 * x["std"])).rjust(15))
    print(t.ljust(17) + "".join(cells), row["shield"]["k"], row["shield"]["params"])
print()
for t, row in r["kd"].items():
    v1 = row["v1"]
    print(t.ljust(17), "teacher v1 %.2f v2 %.2f | shield v1 %.2f v2 %.2f | kd_all-scratch v2 %+.2f (v1 %+.2f) | shield-scratch %+.2f | T-S gap v2 %.2f (v1 %.2f) | int8 drop v2 %.2f (v1 %.2f) | agree %s | epochs %s" % (
        100 * v1["teacher"], 100 * row["teacher"]["f1"]["mean"], 100 * v1["shield"], 100 * row["shield"]["f1"]["mean"],
        100 * row["kd_minus_scratch"], 100 * (v1["kd_all"] - v1["scratch_all"]), 100 * row["shield_minus_scratch"],
        100 * row["teacher_minus_shield"], 100 * (v1["teacher"] - v1["shield"]),
        100 * row["shield"]["int8_drop"]["mean"], 100 * v1["shield_int8_drop"],
        row["shield"]["agreement"] and round(row["shield"]["agreement"]["mean"], 4), row["teacher"]["epochs_run"]))
print("shield vs kd_shap", r["shield_vs_kd_shap"])
print("KD CD", r["kd_cd"])
print("PL CD", r["pl_cd"])
print("gap", {k: v for k, v in r["pl_gap"].items() if k in ("hybrid_eho_aco", "hybrid_no_ants", "hybrid_no_ls", "hybrid_swap_ls", "sa", "aco", "eho", "ga", "pso", "random")})
print("zero-gap share", r["pl_zero_share"])
for k, v in r["pl_large"].items(): print("large", k, list(v.items())[:6])
print("runtime", {k: v for k, v in r["pl_runtime"].items()})
print("fewshot not better", r["fewshot_not_better"], "/", r["fewshot_cells"])
print("shap agree", {k: {kk: vv for kk, vv in v.items() if not kk.startswith("top10_x")} for k, v in r["shap_agreement"].items()})
for t, v in r["latency"].items(): print("lat", t, int(v["teacher"]["throughput_b256"]), int(v["shield"]["throughput_b256"]), int(v["shield_int8"]["throughput_b256"]), v["shield"]["size_kb"], v["shield_int8"]["size_kb"], v["shield"]["params"], v["shield"]["n_features"])
