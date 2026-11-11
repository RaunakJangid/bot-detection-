"""P2 v4, T4: escalation policies compared on the real shifted streams and under flooding (data of scripts/85, 88).

Policies (10% budget; W = 300; refresh every 50 flows; quota q = 30 escalations per 300 flows):
  fixed       validation 90th-percentile threshold, unchanged at deployment
  quota_only  fixed threshold plus the admission quota
  quantile    online running quantile (the cap of scripts/85)
  both        online quantile plus quota (the rule of scripts/88)
Streams: the shifted test streams of scripts/85 (CICIoMT2024 6-class and 19-class, IoT-23 1-1 and 49-1), and the
flooding injection of scripts/88 (spread and burst, 0.25 / 1.0 / 3.2 injected per genuine flow) on CICIoMT2024 6-class
and IoT-23 1-1. 5 student seeds. Per run: overall escalated share, worst and 95th-percentile 1,000-flow block share,
macro-F1 (genuine flows), and the number of threshold changes (adaptation activity).
Output: outputs_v5/p2/policy_compare.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m85, m88 = _l("85_p2v4_cap_quality.py"), _l("88_p2v4_escalation_attack.py")
W, Q = 300, 30


def quota_on(base):
    esc = np.zeros(len(base), bool); hist = np.zeros(len(base) + 1, int)
    for i in range(len(base)):
        if base[i] and hist[i] - hist[max(0, i - W)] < Q:
            esc[i] = True
        hist[i + 1] = hist[i] + esc[i]
    return esc


def policies(H, tv):
    fixed = H > tv; quant = m85.online(H, tv)
    return {"fixed": fixed, "quota_only": quota_on(fixed), "quantile": quant, "both": quota_on(quant)}


def summarise(esc, y, pe, ps, g=None):
    bl = m85.blocks(esc); e_g = esc if g is None else esc[g]
    return {"e": float(esc.mean()), "block_max": float(max(bl)) if bl else None,
            "block_p95": float(np.quantile(bl, .95)) if bl else None, "f1": m85.mf1(y, np.where(e_g, pe, ps))}


def main():
    out = {"streams": [], "flood": []}
    keep = {"ciciomt_category", "ciciomt_attack", "iot23_1-1"}
    from shield.utils.config import Paths
    Pp = Paths(False)
    for st, seed, (y, pe, ps, Hv, Ht) in m88.streams(Pp):
        tv = np.quantile(Hv, 0.9)
        for pol, esc in policies(Ht, tv).items():
            out["streams"].append({"setting": st, "seed": seed, "policy": pol, **summarise(esc, y, pe, ps)})
        if st in ("ciciomt_category", "iot23_1-1"):
            n = len(y); top = np.flatnonzero(Ht >= np.quantile(Ht, 0.9))
            for rho in (0.25, 1.0, 3.2):
                for pattern in ("spread", "burst"):
                    rng = np.random.default_rng(1000 * seed + int(100 * rho) + (pattern == "burst"))
                    k = int(round(rho * n)); src = rng.choice(top, k, replace=True); mask = m88.inject(n, k, pattern, rng)
                    H = np.empty(n + k); H[~mask] = Ht; H[mask] = Ht[src]
                    for pol, esc in policies(H, tv).items():
                        out["flood"].append({"setting": st, "seed": seed, "rho": rho, "pattern": pattern, "policy": pol,
                                             **summarise(esc, y, pe, ps, ~mask)})
        print(st, seed, flush=True)
    summ = {}
    for part in ("streams", "flood"):
        groups = {}
        for r in out[part]:
            k = (r["setting"], r.get("rho"), r.get("pattern"), r["policy"])
            groups.setdefault(k, []).append(r)
        summ[part] = {"|".join(map(str, k)): {m: float(np.median([x[m] for x in v if x[m] is not None]))
                                              for m in ("e", "block_max", "block_p95", "f1")} for k, v in groups.items()}
    out["summary"] = summ
    json.dump(out, open(OUT / "policy_compare.json", "w"), indent=1)
    for part in summ:
        for k, v in summ[part].items():
            print(part, k, {m: round(x, 3) for m, x in v.items()})
    print("saved")


if __name__ == "__main__":
    main()
