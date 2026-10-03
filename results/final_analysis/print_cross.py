import collections
import json

r = json.load(open(r"D:\shield_run\outputs\final_analysis\final_results.json"))


def f(v):
    return "-" if v != v or v is None else "%.3f" % v


z = {(x["source"], x["target"], x["task"], x["model"], x["norm"]): x for x in r["zero_shot"]}
print("ZERO-SHOT (mean+-std over 3 seeds)")
for s, t in [("xciciomt", "xciciot"), ("xciciot", "xciciomt")]:
    for task in ["binary", "shared5"]:
        for m in ["teacher", "xgboost", "scratch_all", "kd_all", "kd_shap", "shield"]:
            a, b = z[(s, t, task, m, "source")], z[(s, t, task, m, "target")]
            print(f"  {s}->{t} {task:7s} {m:11s} in={a['in_domain_macro_f1_mean']:.3f} "
                  f"src={a['macro_f1_mean']:.3f}+-{f(a['macro_f1_std'])} tgt={b['macro_f1_mean']:.3f}+-{f(b['macro_f1_std'])}")
print("FEWSHOT not better")
for x in r["fewshot"]:
    if x["ft_minus_scratch"] <= 0:
        print("  ", x["source"], x["target"], x["task"], x["model"], x["fraction"], round(x["mean_finetuned"], 4), round(x["mean_scratch"], 4))
print("FEWSHOT shield")
for x in r["fewshot"]:
    if x["model"] == "shield":
        print("  ", x["source"], x["target"], x["task"], x["fraction"],
              "ft %.3f sc %.3f d %+.3f" % (x["mean_finetuned"], x["mean_scratch"], x["ft_minus_scratch"]))
u = collections.defaultdict(dict)
for x in r["unseen"]:
    u[(x["source"], x["target"], x["family"])][f"{x['model']}/{x['norm']}"] = round(x["mean"], 3)
for k, v in u.items():
    print("UNSEEN", k, v)
print("INFLATION")
for x in r["inflation"]:
    print("  ", x)
