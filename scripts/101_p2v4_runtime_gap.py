"""P2 v4, T5: EHO-ACO and the other search methods against the exact ILP, in runtime and quality, from existing logs.

Small FFCCP instances with a proven optimum (outputs_v3/killi_rao/optimum.json, ILP seconds) and the heuristic runs of
outputs_v3/killi_rao/runs.jsonl (z and seconds per run, 10 runs, 20,000 evaluations). Large Cogentco/Kdl settings:
outputs_v3/killi_rao/large_runs.jsonl (seconds per run, 30 runs) and the X8 quality (outputs_v5/p2/search_runs.jsonl).
Per method: median and 95th-percentile runtime per run, mean gap to the optimum and optimum-hit share on small
instances; the ILP's median and maximum solve time on the same instances; runtime ratio ILP / method.
Output: outputs_v5/p2/runtime_gap.json."""
import json
from collections import defaultdict

import numpy as np

V3 = "D:/shield_run/outputs_v3/killi_rao"
opt = json.load(open(f"{V3}/optimum.json"))
small = [json.loads(l) for l in open(f"{V3}/runs.jsonl")]
large = [json.loads(l) for l in open(f"{V3}/large_runs.jsonl")]
key = lambda r: f"{r['topology']}|{r.get('K', 2)}|{r['p']}"
proven = {k: v for k, v in opt.items() if v.get("status") == 0}
out = {"ilp": {"n_proven": len(proven), "median_s": float(np.median([v["seconds"] for v in proven.values()])),
               "max_s": float(np.max([v["seconds"] for v in proven.values()]))}, "small": {}, "large": {}}
g = defaultdict(list)
for r in small:
    if key(r) in proven:
        g[r["alg"]].append(r)
for alg, R in sorted(g.items()):
    gaps = [100 * (r["z"] / proven[key(r)]["z"] - 1) for r in R if r["z"] < 1000]
    secs = [r["seconds"] for r in R]
    ilp = [proven[key(r)]["seconds"] for r in R]
    out["small"][alg] = {"runs": len(R), "median_s": float(np.median(secs)), "p95_s": float(np.quantile(secs, .95)),
                         "mean_gap_pct": float(np.mean(gaps)) if gaps else None,
                         "opt_hit": float(np.mean([abs(r["z"] - proven[key(r)]["z"]) <= 1e-6 * max(1, proven[key(r)]["z"]) for r in R])),
                         "ilp_over_method_median": float(np.median(np.array(ilp) / np.maximum(secs, 1e-9)))}
gl = defaultdict(list)
for r in large:
    gl[r["alg"]].append(r["seconds"])
for alg, s in sorted(gl.items()):
    out["large"][alg] = {"runs": len(s), "median_s": float(np.median(s)), "p95_s": float(np.quantile(s, .95))}
json.dump(out, open("D:/shield_run/outputs_v5/p2/runtime_gap.json", "w"), indent=1)
print("ILP", out["ilp"])
for part in ("small", "large"):
    for alg, v in out[part].items():
        print(part, alg, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
