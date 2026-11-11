"""v4: summarise the matched live ablation (outputs_v4/liveblock/<scenario>/<config>_r<k>.json from e7v4_liveblock.py).

Per scenario and configuration over the repeats: for each attack event, runs in which its source(s) were blocked, time
to block (median, range) and share of attack packets dropped; normal hosts wrongly blocked (runs with any, hosts per
run); KS warning: share of checks above the Eq. (9) critical value and time from the first event to the first warning.
Bursts: per event, runs in which any normal source of the event was blocked during it. Distributed: low-rate sources
blocked and spoofed sources blocked. Output: outputs_v4/live_v4_summary.json."""

import glob
import json
from pathlib import Path

import numpy as np

D = Path("D:/shield_run/outputs_v4/liveblock")
CONFIGS = ["lab", "commissioned", "rate_always", "ks_rate", "full"]


def med_rng(v):
    v = [x for x in v if x is not None]
    return None if not v else {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}


def main():
    out = {}
    for scen in ("attacks", "bursts", "distributed"):
        out[scen] = {}
        for cfg in CONFIGS:
            runs = [json.load(open(f)) for f in sorted(glob.glob(str(D / scen / f"{cfg}_r[1-9].json")))]
            if not runs:
                continue
            s = {"runs": len(runs), "events": {}}
            for ev in runs[0]["events"]:
                E = [r["events"][ev] for r in runs]
                blocked = [bool(e["blocked_at_s"]) and e["time_to_block_s"] is not None for e in E]
                drop = [e["packets_dropped"] / e["packets_sent"] for e in E if e.get("packets_sent")]
                s["events"][ev] = {"runs_blocked": int(sum(blocked)),
                                   "time_to_block": med_rng([e["time_to_block_s"] for e in E]),
                                   "dropped_share": med_rng(drop),
                                   "warning_during": int(sum(e["warning_during_event"] for e in E)),
                                   "flows_flagged_commissioned": med_rng([e["flows_flagged_commissioned"] for e in E]),
                                   "flows_from_sources": med_rng([e["flows_from_sources"] for e in E])}
            wb = [len(r["benign"]["wrongly_blocked"]) for r in runs]
            s["benign_hosts_blocked"] = {"runs_with_any": int(sum(x > 0 for x in wb)), "per_run": med_rng(wb),
                                         "total": int(sum(wb))}
            s["benign_dropped_packets"] = med_rng([r["benign"]["dropped_packets"] for r in runs])
            s["benign_flagged_commissioned"] = med_rng([r["benign"]["flagged_share_commissioned"] for r in runs])
            s["warning_share_of_checks"] = med_rng([r["warning_share_of_checks"] for r in runs])
            first = []
            for r in runs:
                t0 = min(e["start_s"] for e in r["events"].values())
                w = [t for t, d, cr in r["ks_log"] if d > cr]
                first.append((min(w) - t0) if w else None)
            s["first_warning_after_first_event_s"] = med_rng(first)
            s["warning_before_first_event_runs"] = int(sum(1 for r in runs if any(
                d > cr and t < min(e["start_s"] for e in r["events"].values()) for t, d, cr in r["ks_log"])))
            s["rate_threshold"] = med_rng([r["rate_threshold_flows_per_5s"] for r in runs])
            s["n_warm_flows"] = med_rng([r["n_warm_flows"] for r in runs])
            s["thr_lab"] = med_rng([r["thresholds"]["lab"] for r in runs])
            s["thr_commissioned"] = med_rng([r["thresholds"]["commissioned"] for r in runs])
            s["ks_checks_before_first_event"] = int(sum(1 for r in runs for t, d, cr in r["ks_log"]
                                                        if t < min(e["start_s"] for e in r["events"].values())))
            # each block attributed to the event window in which it happened (start <= t < end), with its rule
            bw = {}
            for r in runs:
                for ip, t in r["blocked_at"].items():
                    ev = [k for k, e in r["events"].items() if e["start_s"] <= t < e["end_s"]]
                    key = f"{ev[0] if ev else 'outside'}:{r['block_reason'][ip]}:{'normal' if ip in r['benign']['wrongly_blocked'] else 'attack'}"
                    bw[key] = bw.get(key, 0) + 1
            s["blocks_by_window"] = bw
            if scen == "distributed":
                s["spoofed_sources_blocked"] = med_rng([r.get("spoofed_sources_blocked") for r in runs])
                s["spoofed_flows_seen"] = med_rng([r.get("spoofed_flows_seen") for r in runs])
            out[scen][cfg] = s
            print(scen, cfg, "runs", len(runs), "benign blocked", s["benign_hosts_blocked"],
                  {ev: (v["runs_blocked"], v["time_to_block"] and round(v["time_to_block"]["median"], 1))
                   for ev, v in s["events"].items()})
    json.dump(out, open("D:/shield_run/outputs_v4/live_v4_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
