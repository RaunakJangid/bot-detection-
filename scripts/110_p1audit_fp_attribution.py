"""P1 audit (read-only, no P1 file changed): where do the lab-threshold false alarms of P1's IoT-23 settings fall?

Inputs: P1's per-flow scores outputs_v4/regime/iot23_*.npz (scripts/66) and the IoT-23 sample (scripts/41). Every
deployment flow is joined to its sample row by (capture, ts); the join is unique and all labels agree (checked here).
Threshold and FPR exactly as scripts/70: t0 = 99th percentile of the model's validation scores on benign flows (A = 1%),
false alarm = benign deployment flow with score > t0.
Pre-specified candidate category (same definition as the capture audit of 2026-11-08): benign-labelled flow that
originates from the capture's infected host (most frequent originator among the capture's Malicious flows in the whole
sample), protocol TCP, connection state in {S0, REJ, RSTO, RSTOS0, SH} (connection attempt without an established session).
This is a candidate label-semantics category, not a confirmed labelling error; the original labels are kept.
Per setting and model: FPR on all benign flows (reproduces exp12_v4.json), share of false alarms on candidate flows,
FPR on non-candidate benign flows (sensitivity analysis), recall unchanged. Medians and ranges over models.
Output: outputs_v5/p1_audit/fp_attribution.json."""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

A = 0.01
SCAN_STATES = {"S0", "REJ", "RSTO", "RSTOS0", "SH"}
OUT = Path("D:/shield_run/outputs_v5/p1_audit")


def main():
    from shield.utils.config import Paths
    P = Paths(False); OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet", columns=["capture", "ts", "y", "oh", "rp", "proto", "conn_state"])
    bot = df[df.y == 1].groupby("capture").oh.agg(lambda s: s.value_counts().index[0]).to_dict()
    df["cand"] = (df.y == 0) & (df.oh == df.capture.map(bot)) & (df.proto == "tcp") & df.conn_state.isin(SCAN_STATES)
    E = json.load(open("D:/shield_run/outputs_v4/exp12_v4.json")); res = {}
    for f in sorted(glob.glob("D:/shield_run/outputs_v4/regime/iot23*.npz")):
        s = os.path.basename(f)[:-4]; Z = np.load(f, allow_pickle=True)
        m = pd.DataFrame({"capture": Z["cap"], "ts": Z["ts"]}).merge(df, on=["capture", "ts"], how="left")
        yt = Z["yt"].astype(bool); yv = Z["yv"].astype(bool)
        assert m.y.notna().all() and (m.y.values.astype(bool) == yt).all(), s
        cand = m.cand.values.astype(bool); ben = ~yt
        rows = []
        for i in range(len(Z["models"])):
            sv, st = Z["sv"][i].astype(np.float64), Z["st"][i].astype(np.float64)
            t0 = np.quantile(sv[~yv], 1 - A); fp = ben & (st > t0)
            rows.append({"fpr_all": float(fp.sum() / ben.sum()),
                         "fp_share_on_candidate": float((fp & cand).sum() / fp.sum()) if fp.sum() else None,
                         "fpr_candidate": float((fp & cand).sum() / cand.sum()) if cand.sum() else None,
                         "fpr_noncandidate": float((fp & ~cand).sum() / (ben & ~cand).sum()) if (ben & ~cand).sum() else None,
                         "recall": float((yt & (st > t0)).sum() / yt.sum()) if yt.sum() else None})
        med = lambda k: (float(np.median([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None)
        rng = lambda k: ([float(min(r[k] for r in rows if r[k] is not None)), float(max(r[k] for r in rows if r[k] is not None))]
                         if any(r[k] is not None for r in rows) else None)
        res[s] = {"n_benign": int(ben.sum()), "n_candidate_benign": int(cand.sum()), "candidate_share_of_benign": float(cand.sum() / ben.sum()),
                  "n_models": len(rows), "fpr_all_median": med("fpr_all"), "exp12_fpr_lab_median": E[s]["fpr_lab_median"],
                  "reproduces_exp12": bool(abs(med("fpr_all") - E[s]["fpr_lab_median"]) < 1e-9),
                  **{f"{k}_median": med(k) for k in ("fp_share_on_candidate", "fpr_candidate", "fpr_noncandidate", "recall")},
                  **{f"{k}_range": rng(k) for k in ("fp_share_on_candidate", "fpr_noncandidate")}, "per_model": rows}
        r = res[s]
        fmt = lambda v, d=1: "-" if v is None else f"{100 * v:.{d}f}"
        print(f"{s[6:58]:52s} benign {r['n_benign']:6d} cand {fmt(r['candidate_share_of_benign'])}% | FPR all {fmt(r['fpr_all_median'])}% "
              f"(exp12 ok {r['reproduces_exp12']}) | FP on cand {fmt(r['fp_share_on_candidate_median'])}% | FPR non-cand "
              f"{fmt(r['fpr_noncandidate_median'], 2)}% range {[fmt(x, 2) for x in (r['fpr_noncandidate_range'] or [])]}", flush=True)
    json.dump(res, open(OUT / "fp_attribution.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
