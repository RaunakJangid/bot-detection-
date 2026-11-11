"""v5: summarise the live additions (outputs_v5/liveblock from e7v5_liveblock.py). Output: outputs_v5/live_v5_summary.json.

B1 benign-only scenario (no event after the 60 s window, 300 s of normal traffic), 5 configurations x 5 runs:
   KS checks per run, share of checks above the Eq. (9) critical value, time of the first warning after the window,
   normal hosts blocked (and by which rule), Step 1 threshold, window size. Because the warning is computed the same
   way in every configuration, the warning statistics are pooled over all 25 runs.
B3 storm sweep (rate rule always on): all six normal hosts send n HTTP requests spaced g seconds (about 10 s);
   per run: rate limit, the largest per-host count of new flows in any 5 s window during the storm (from the logged
   flows), hosts blocked. Reported per spacing g: peak / limit ratio and hosts blocked."""

import glob
import json
from pathlib import Path

import numpy as np

D = Path("D:/shield_run/outputs_v5/liveblock")
CFG = ["lab", "commissioned", "rate_always", "ks_rate", "full"]
BENIGN = [f"10.0.0.{i}" for i in range(1, 7)]
WARM = 60


def mr(v):
    v = [x for x in v if x is not None]
    return None if not v else {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}


out = {"benign": {}, "storm": {}}
allchecks, firsts = [], []
for cfg in CFG:
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(D / "benign" / f"{cfg}_r[1-9].json")))]
    if not runs:
        continue
    for r in runs:
        allchecks += [d > c for _, d, c in r["ks_log"]]
        w = [t for t, d, c in r["ks_log"] if d > c]
        firsts.append(min(w) - WARM if w else None)
    out["benign"][cfg] = {
        "runs": len(runs), "checks_per_run": mr([len(r["ks_log"]) for r in runs]),
        "warning_share": mr([r["warning_share_of_checks"] for r in runs]),
        "runs_with_any_warning": int(sum(any(d > c for _, d, c in r["ks_log"]) for r in runs)),
        "normal_hosts_blocked_total": int(sum(len(r["benign"]["wrongly_blocked"]) for r in runs)),
        "runs_with_any_block": int(sum(len(r["benign"]["wrongly_blocked"]) > 0 for r in runs)),
        "block_reasons": {k: sum(1 for r in runs for ip, why in r["block_reason"].items() if why == k) for k in ("model", "rate")},
        "n_warm_flows": mr([r["n_warm_flows"] for r in runs]), "thr_commissioned": mr([r["thresholds"]["commissioned"] for r in runs]),
        "rate_threshold": mr([r["rate_threshold_flows_per_5s"] for r in runs]),
        "flagged_share_commissioned": mr([r["benign"]["flagged_share_commissioned"] for r in runs]),
        "n_flows_after_window": mr([r["benign"]["n_flows_after_warm"] for r in runs])}
out["benign_pooled"] = {"checks": len(allchecks), "warning_share": float(np.mean(allchecks)) if allchecks else None,
                        "runs": len(firsts), "runs_with_any_warning": int(sum(f is not None for f in firsts)),
                        "first_warning_after_window_s": mr(firsts)}

for sdir in sorted(glob.glob(str(D / "storm_g*"))):
    name = Path(sdir).name; g = int(name[-3:]) / 100
    rows = []
    for f in sorted(glob.glob(f"{sdir}/rate_always_r[1-9].json")):
        r = json.load(open(f)); fl = json.load(open(f.replace("rate_always_r", "flows_rate_always_r")))
        ev = r["events"]["storm"]; t0, t1 = ev["start_s"], ev["start_s"] + 15
        peak = max([x["n5"] for x in fl if x["src"] in BENIGN and t0 <= x["t"] < t1] or [0])
        lim = r["rate_threshold_flows_per_5s"]
        rows.append({"limit": lim, "peak": peak, "ratio": peak / lim, "hosts_blocked": len(r["benign"]["wrongly_blocked"]),
                     "time_to_block": ev["time_to_block_s"]})
    out["storm"][name] = {"gap_s": g, "requests_per_host": int(round(10 / g)), "runs": len(rows),
                          "limit": mr([x["limit"] for x in rows]), "peak": mr([x["peak"] for x in rows]),
                          "ratio": mr([x["ratio"] for x in rows]), "hosts_blocked": [x["hosts_blocked"] for x in rows],
                          "hosts_blocked_total": int(sum(x["hosts_blocked"] for x in rows)),
                          "time_to_block": mr([x["time_to_block"] for x in rows]), "per_run": rows}
json.dump(out, open("D:/shield_run/outputs_v5/live_v5_summary.json", "w"), indent=1)
print(json.dumps(out["benign_pooled"], indent=1))
for k, v in out["benign"].items():
    print(k, {x: v[x] for x in ("runs", "warning_share", "runs_with_any_warning", "normal_hosts_blocked_total", "block_reasons")})
for k, v in out["storm"].items():
    print(k, v["gap_s"], "ratio", v["ratio"], "blocked", v["hosts_blocked"])
