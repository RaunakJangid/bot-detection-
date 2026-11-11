"""P1 audit item P1-01 (read-only, no P1 result file changed): is the Case 1 / Case 2 split of P1's IoT-23 settings a
label artefact?

R*_1% as scripts/70 (oracle recall at <= 1% deployment FPR, Case 2 if the median over models is < 0.5), computed twice:
  all       every benign deployment flow (must reproduce rstar_median of outputs_v4/exp12_v4.json)
  no_cand   without the candidate benign flows of scripts/110 (connection attempts of the capture's infected host that
            are labelled benign: TCP, state S0/REJ/RSTO/RSTOS0/SH); attacks unchanged
The second is a sensitivity analysis, not a relabelling: it shows how far each setting's case depends on those flows.
Output: outputs_v5/p1_audit/case2_label_sensitivity.json."""
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("D:/shield_run/outputs_v5/p1_audit/case2_label_sensitivity.json")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m70, m110 = _l("70_v4_exp12.py"), _l("110_p1audit_fp_attribution.py")


def main():
    from shield.utils.config import Paths
    P = Paths(False)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet", columns=["capture", "ts", "y", "oh", "proto", "conn_state"])
    bot = df[df.y == 1].groupby("capture").oh.agg(lambda s: s.value_counts().index[0]).to_dict()
    df["cand"] = (df.y == 0) & (df.oh == df.capture.map(bot)) & (df.proto == "tcp") & df.conn_state.isin(m110.SCAN_STATES)
    E = json.load(open("D:/shield_run/outputs_v4/exp12_v4.json")); res = {}
    for f in sorted(glob.glob("D:/shield_run/outputs_v4/regime/iot23*.npz")):
        s = os.path.basename(f)[:-4]; Z = np.load(f, allow_pickle=True)
        m = pd.DataFrame({"capture": Z["cap"], "ts": Z["ts"]}).merge(df, on=["capture", "ts"], how="left")
        yt = Z["yt"].astype(bool)
        assert m.y.notna().all() and (m.y.values.astype(bool) == yt).all(), s
        cand = m.cand.values.astype(bool)
        ra, rn = [], []
        for i in range(len(Z["models"])):
            st = Z["st"][i].astype(np.float64)
            ra.append(m70.rstar(st[~yt], st[yt])); rn.append(m70.rstar(st[~yt & ~cand], st[yt]))
        a, n = float(np.median(ra)), float(np.median(rn))
        res[s] = {"n_benign": int((~yt).sum()), "n_candidate_benign": int(cand.sum()), "rstar_all_median": a,
                  "exp12_rstar_median": E[s]["rstar_median"], "reproduces_exp12": bool(abs(a - E[s]["rstar_median"]) < 1e-9),
                  "rstar_no_cand_median": n, "rstar_no_cand_range": [float(min(rn)), float(max(rn))],
                  "case2_all": a < 0.5, "case2_no_cand": n < 0.5, "case_changes": (a < 0.5) != (n < 0.5)}
        r = res[s]
        print(f"{s[6:58]:52s} cand {r['n_candidate_benign']:5d}/{r['n_benign']:6d} | R* all {a:.2f} (ok {r['reproduces_exp12']}) "
              f"| no cand {n:.2f} [{min(rn):.2f}, {max(rn):.2f}] | case change {r['case_changes']}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
