"""Corrected E8 criterion: recall loss vs the ORACLE threshold at the same realised FPR (best recall achievable by any
threshold on that capture at that false-alarm rate), instead of vs the fixed threshold (which fires on 11-74% of
benign flows and so penalises any FPR reduction). Eval rows = capture flows after the first K benign (time order)."""
import json
import sys

import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
tag = "" if K == 1000 else f"_k{K}"
E = json.load(open(rf"D:\shield_run\outputs_v3\iot23\e8_stream{tag}.json"))
for design in ("same_family", "capture_shift"):
    Z = np.load(rf"D:\shield_run\outputs_v3\iot23\scores_{design}.npz")
    models = list(Z["models"]); st, yt, cap, ts = Z["st"], Z["yt"], Z["capture"], Z["ts"]
    D = E[design]
    print(f"\n== {design} (K={K}) captures {D['captures']}")
    for meth in ("fixed", "burnin_rand", "burnin_time", "roll_free", "roll_audit", "nguyen"):
        fpr_m, loss_m, ok = [], [], []
        for mi, m in enumerate(models):
            fp, ls = [], []
            for r in D["models"][m][meth]["per_capture"]:
                idx = np.flatnonzero(cap == r["capture"]); idx = idx[np.argsort(ts[idx], kind="stable")]
                y = yt[idx]; s = st[mi][idx]
                ev = np.ones(len(y), bool); ev[np.flatnonzero(y == 0)[:K]] = False
                sb, sa = np.sort(s[ev & (y == 0)]), s[ev & (y == 1)]
                if len(sa) == 0:
                    continue
                t = sb[min(len(sb) - 1, int(np.ceil((1 - r["fpr"]) * len(sb))) - 1)] if r["fpr"] < 1 else -np.inf
                oracle = float((sa > t).mean())
                fp.append(r["fpr"]); ls.append(oracle - (r["recall"] or 0))
            if fp:
                fpr_m.append(np.mean(fp)); loss_m.append(np.mean(ls))
        fpr_m, loss_m = np.array(fpr_m), np.array(loss_m)
        rule = np.median(fpr_m) <= 0.015 and np.median(loss_m) <= 0.05
        print(f"  {meth:12s} FPR med {100*np.median(fpr_m):6.2f}% | recall loss vs oracle@same FPR med {100*np.median(loss_m):+6.1f} | rule {'PASS' if rule else 'fail'}")
