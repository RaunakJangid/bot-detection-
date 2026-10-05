import json, collections
r = json.load(open(r"D:\shield_run\outputs_v2\final_analysis\v2_results.json"))
z = collections.defaultdict(dict)
for x in r["zero_shot"]:
    z[(x["source"], x["target"], x["task"], x["model"])][x["norm"]] = (x["macro_f1_mean"], x.get("macro_f1_std"), x["in_domain_macro_f1_mean"])
norms = ["source", "target", "coral", "dann"]
wins = collections.Counter()
for k, v in sorted(z.items()):
    present = {n: v[n][0] for n in norms if n in v}
    best = max(present, key=present.get)
    wins[best] += 1
    print(k, "in=%.3f" % v["source"][2], " ".join("%s=%.3f" % (n, present[n]) for n in norms if n in present), "best:", best)
print("best-normalisation counts:", dict(wins))
for n in ["coral", "dann"]:
    better_src = sum(1 for v in z.values() if n in v and v[n][0] > v["source"][0])
    better_tgt = sum(1 for v in z.values() if n in v and v[n][0] > v["target"][0])
    tot = sum(1 for v in z.values() if n in v)
    print(n, "beats source in", better_src, "/", tot, "; beats target in", better_tgt, "/", tot)
u = collections.defaultdict(dict)
for x in r["unseen"]:
    u[(x["source"], x["target"], x["family"])][x["model"] + "/" + x["norm"]] = x["mean"]
for k, v in u.items():
    print("UNSEEN", k, {m: round(s, 3) for m, s in v.items() if m.startswith(("shield", "xgboost", "kd_all"))})
