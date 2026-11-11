"""P1 correction check: share of live KS-warning checks with and without IPv6 link-local flows (read-only replay).

The live gateway (scripts/e7v4_liveblock.py, e7v5_liveblock.py) scored every non-server Zeek flow, used the flows of the
first WARM = 60 s as the commissioning window, and every 50 post-window flows compared the last 300 scores (at least 100)
with the window scores by a two-sample KS test at alpha = 0.001 (critical 1.949 sqrt((n1+n2)/(n1 n2))).
This script replays that rule on the stored flow logs (flows_<config>_r<rep>.json) twice:
  all      every logged flow (the gateway's own stream; must reproduce warning_share_of_checks of each run exactly)
  ipv4     the same stream without IPv6 link-local sources (fe80::/10, neighbour discovery)
and, for the gated configuration (ks_rate), whether the IPv4-only warning was also on at the moment each host was blocked
(counterfactual valid up to the first block: later flows of a blocked host are missing from the log).
Output: outputs_v5/p1_live_ipv6.json."""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

WARM, OUT = 60.0, Path("D:/shield_run/outputs_v5/p1_live_ipv6.json")
DIRS = {"v4": "D:/shield_run/outputs_v4/liveblock", "v5": "D:/shield_run/outputs_v5/liveblock"}


def ks(a, b):
    a, b = np.sort(a), np.sort(b); v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def replay(flows):
    """[(t, warning_on)] at every check, exactly as the gateway computes it."""
    warm = np.array([f["score"] for f in flows if f["t"] < WARM]); rec, n, out = [], 0, []
    for f in flows:
        if f["t"] < WARM:
            continue
        rec.append(f["score"]); rec = rec[-300:]; n += 1
        if n % 50 == 0 and len(rec) >= 100 and len(warm):
            out.append((f["t"], ks(np.array(rec), warm) > 1.949 * np.sqrt((len(rec) + len(warm)) / (len(rec) * len(warm)))))
    return out


def state_at(checks, t):
    on = False
    for tc, w in checks:
        if tc > t:
            break
        on = w
    return on


def main():
    runs, agg = [], defaultdict(lambda: {"all": [], "ipv4": [], "ipv6_share": [], "repro_ok": 0, "n": 0})
    for ver, root in DIRS.items():
        for scen in sorted(os.listdir(root)):
            for f in sorted(glob.glob(f"{root}/{scen}/flows_*_r*.json")):
                cfg, rep = os.path.basename(f)[6:-5].rsplit("_r", 1)
                summ_f = f"{root}/{scen}/{cfg}_r{rep}.json"
                if not os.path.exists(summ_f):
                    continue
                flows = json.load(open(f)); S = json.load(open(summ_f))
                flows = flows["flows"] if isinstance(flows, dict) else flows
                ca = replay(flows); c4 = replay([x for x in flows if not str(x["src"]).lower().startswith("fe80")])
                share_all = float(np.mean([w for _, w in ca])) if ca else None
                share_v4 = float(np.mean([w for _, w in c4])) if c4 else None
                orig = S.get("warning_share_of_checks")
                ok = orig is not None and share_all is not None and abs(orig - share_all) < 1e-9
                r = {"ver": ver, "scenario": scen, "config": cfg, "rep": int(rep), "n_flows": len(flows),
                     "ipv6_share": float(np.mean([str(x["src"]).lower().startswith("fe80") for x in flows])),
                     "n_checks_all": len(ca), "share_all": share_all, "n_checks_ipv4": len(c4), "share_ipv4": share_v4,
                     "orig_share": orig, "reproduces_orig": ok}
                if cfg == "ks_rate":
                    blocks = S.get("blocked_at") or {}
                    r["gated_blocks"] = {h: {"t": t, "reason": (S.get("block_reason") or {}).get(h),
                                             "warning_all_on": state_at(ca, t), "warning_ipv4_on": state_at(c4, t)}
                                         for h, t in blocks.items()}
                runs.append(r); a = agg[(ver, scen)]
                a["all"] += [w for _, w in ca]; a["ipv4"] += [w for _, w in c4]; a["ipv6_share"].append(r["ipv6_share"])
                a["repro_ok"] += ok; a["n"] += 1
    summary = {f"{v}|{s}": {"runs": a["n"], "reproduced_runs": a["repro_ok"], "pooled_share_all": float(np.mean(a["all"])) if a["all"] else None,
                            "n_checks_all": len(a["all"]), "pooled_share_ipv4": float(np.mean(a["ipv4"])) if a["ipv4"] else None,
                            "n_checks_ipv4": len(a["ipv4"]), "ipv6_flow_share_median": float(np.median(a["ipv6_share"]))}
               for (v, s), a in agg.items()}
    gated = [(r["scenario"], r["rep"], h, b) for r in runs if r["config"] == "ks_rate" for h, b in r.get("gated_blocks", {}).items()]
    summary["_gated_blocks"] = {"n": len(gated), "warning_all_on_at_block": sum(b["warning_all_on"] for *_, b in gated),
                                "warning_ipv4_on_at_block": sum(b["warning_ipv4_on"] for *_, b in gated),
                                "by_scenario": {s: {"n": sum(1 for g in gated if g[0] == s),
                                                    "ipv4_on": sum(1 for g in gated if g[0] == s and g[3]["warning_ipv4_on"])}
                                                for s in sorted({g[0] for g in gated})}}
    json.dump({"summary": summary, "runs": runs}, open(OUT, "w"), indent=1, default=lambda o: o.item())
    for k, v in summary.items():
        print(k, v)
    print("saved", OUT)


if __name__ == "__main__":
    main()
