"""P1 audit item P1-01, part 2 (read-only): does "Case 2 is not repaired by Step 1" depend on the suspect labels?

Step 1 exactly as scripts/70 for the time-ordered IoT-23 settings: t_c = split-conformal 1% quantile of the first K benign
deployment flows in time order (K = 1,000, or half the benign flows if fewer than 2,000); FPR on the remaining benign
flows, recall on all attacks. Computed twice:
  all       every benign flow (must reproduce fpr_com_median / recall_com_median of outputs_v4/exp12_v4.json)
  no_cand   candidate benign flows of scripts/110 removed from both the window and the evaluation (K re-derived)
Sensitivity analysis only; labels are not changed. Output: outputs_v5/p1_audit/step1_label_sensitivity.json."""
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("D:/shield_run/outputs_v5/p1_audit/step1_label_sensitivity.json")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m70, m110 = _l("70_v4_exp12.py"), _l("110_p1audit_fp_attribution.py")


def step1(st, ts, ben, att):
    o = np.argsort(ts, kind="stable"); ob = o[ben[o]]
    K = 1000 if len(ob) >= 2000 else len(ob) // 2
    tc = m70.conformal(st[ob[:K]])
    return float((st[ob[K:]] > tc).mean()), float((st[att] > tc).mean()), K, len(ob) - K


def main():
    from shield.utils.config import Paths
    P = Paths(False)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet", columns=["capture", "ts", "y", "oh", "proto", "conn_state"])
    bot = df[df.y == 1].groupby("capture").oh.agg(lambda s: s.value_counts().index[0]).to_dict()
    df["cand"] = (df.y == 0) & (df.oh == df.capture.map(bot)) & (df.proto == "tcp") & df.conn_state.isin(m110.SCAN_STATES)
    E = json.load(open("D:/shield_run/outputs_v4/exp12_v4.json")); res = {}
    for f in sorted(glob.glob("D:/shield_run/outputs_v4/regime/iot23*.npz")):
        s = os.path.basename(f)[:-4]
        if s == "iot23_random_control_all":
            continue                                   # not time-ordered in scripts/70
        Z = np.load(f, allow_pickle=True)
        m = pd.DataFrame({"capture": Z["cap"], "ts": Z["ts"]}).merge(df, on=["capture", "ts"], how="left")
        yt = Z["yt"].astype(bool); assert m.y.notna().all() and (m.y.values.astype(bool) == yt).all(), s
        cand = m.cand.values.astype(bool); ts = np.asarray(Z["ts"], np.float64)
        A, N = [], []
        for i in range(len(Z["models"])):
            st = Z["st"][i].astype(np.float64)
            A.append(step1(st, ts, ~yt, yt)); N.append(step1(st, ts, ~yt & ~cand, yt))
        med = lambda R, j: float(np.median([r[j] for r in R]))
        res[s] = {"fpr_all": med(A, 0), "recall_all": med(A, 1), "K_all": A[0][2], "n_eval_all": A[0][3],
                  "reproduces_exp12": bool(abs(med(A, 0) - E[s]["fpr_com_median"]) < 1e-9
                                           and abs(med(A, 1) - E[s]["recall_com_median"]) < 1e-9),
                  "fpr_no_cand": med(N, 0), "recall_no_cand": med(N, 1), "K_no_cand": N[0][2], "n_eval_no_cand": N[0][3],
                  "fpr_no_cand_range": [float(min(r[0] for r in N)), float(max(r[0] for r in N))],
                  "case2_published": bool(E[s]["case2"])}
        r = res[s]
        print(f"{s[6:58]:52s} case2 {r['case2_published']!s:5} | all FPR {100 * r['fpr_all']:5.2f}% rec {100 * r['recall_all']:5.1f}% "
              f"(ok {r['reproduces_exp12']}, K {r['K_all']}) | no cand FPR {100 * r['fpr_no_cand']:5.2f}% rec "
              f"{100 * r['recall_no_cand']:5.1f}% (K {r['K_no_cand']}, eval {r['n_eval_no_cand']})", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
