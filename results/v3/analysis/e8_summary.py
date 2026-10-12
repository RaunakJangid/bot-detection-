"""Summary of E8 (42_e8_stream.py): per design and method, median over models of pooled FPR, median recall, recall loss
vs fixed, share of models holding the budget (pooled FPR <= 1.5%), median hourly-FPR sd and update cost; plus the
decision rule (median model pooled FPR <= 1.5% AND median recall loss <= 5 points)."""
import json
import sys

import numpy as np

for path in sys.argv[1:] or [r"D:\shield_run\outputs_v3\iot23\e8_stream.json"]:
    R = json.load(open(path))
    print(f"\n##### {path}")
    for design, D in R.items():
        print(f"\n== {design}: captures {D['captures']}")
        M = D["models"]
        methods = list(next(iter(M.values())).keys())
        for k in methods:
            fpr = np.array([m[k]["pooled_fpr"] for m in M.values()])
            rec = np.array([m[k]["median_recall"] if m[k]["median_recall"] is not None else np.nan for m in M.values()])
            rec0 = np.array([m["fixed"]["median_recall"] if m["fixed"]["median_recall"] is not None else np.nan for m in M.values()])
            loss = rec0 - rec
            sd = [r["hourly_fpr_sd"] for m in M.values() for r in m[k]["per_capture"] if r["hourly_fpr_sd"] is not None]
            cost = np.median([m[k]["us_per_flow"] for m in M.values()])
            ok = np.nanmedian(fpr) <= 0.015 and (np.all(np.isnan(loss)) or np.nanmedian(loss) <= 0.05)
            print(f"  {k:12s} FPR med {100*np.median(fpr):6.2f}% [{100*fpr.min():5.2f},{100*fpr.max():6.2f}] | "
                  f"models <=1.5%: {(fpr <= 0.015).sum():2d}/{len(fpr)} | recall med {100*np.nanmedian(rec):5.1f} "
                  f"(loss {100*np.nanmedian(loss):+5.1f}) | hourly sd {100*np.median(sd) if sd else float('nan'):5.2f} | "
                  f"{cost:7.2f} us/flow | rule {'PASS' if ok else 'fail'}")
