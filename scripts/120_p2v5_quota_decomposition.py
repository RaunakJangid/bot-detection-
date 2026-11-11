"""P2 v5 (audit P2-03): decomposition of the detection effect attributed to "the quota" in P2 v4 Table VI.

scripts/109 compared "escalate all" (fixed validation threshold) with "quota" (moving threshold + quota), which mixes two
changes and a lower escalation volume. Here, on the same flows, seed pairs and metrics as scripts/109 (CIC test sets in the
rng(0) order; 5 students x 5 teachers; macro-F1 and accuracy), five policies:
  fixed       candidates above the validation 90th-percentile entropy, all escalated
  moving      online 90th percentile of the previous 300 entropies, refreshed every 50 flows (scripts/85 online)
  quota_only  fixed threshold + quota (at most 30 escalations among the previous 300 flows)
  both        moving threshold + quota (the deployed rule, scripts/88 quota; = "quota" of scripts/109)
  equal_vol   fixed rule at the SAME escalated share as `both` for that student: the top-entropy flows of the stream with
              share = admitted share of `both` (separates the volume effect from the ordering/timing effect)
Output: outputs_v5/p2/quota_decomposition.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m85, m88, m109 = _l("85_p2v4_cap_quality.py"), _l("88_p2v4_escalation_attack.py"), _l("109_p2v4_quota_effect_seeds.py")
POL = ("fixed", "moving", "quota_only", "both", "equal_vol")


def quota_on(base, q=30, w=300):
    esc = np.zeros(len(base), bool); hist = np.zeros(len(base) + 1, int)
    for i in range(len(base)):
        if base[i] and hist[i] - hist[max(0, i - w)] < q:
            esc[i] = True
        hist[i + 1] = hist[i] + esc[i]
    return esc


def main():
    from shield.data.datasets import load_processed
    from shield.eval.metrics import softmax
    from shield.utils.config import Paths
    P = Paths(False); res = {}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        yt = load_processed(P.processed(ds), task, splits=("test",)).splits["test"].y.astype(np.int64)
        lc = P.v3 / "logits" / f"{ds}_{task}"; o = np.random.default_rng(0).permutation(len(yt)); y = yt[o]
        PE = [np.load(lc / f"teacher_s{t}_test.npy").argmax(1)[o] for t in range(5)]
        masks, PS = [], []
        for s in range(5):
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))[o]
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            H = m85.ent(Pt); tv = np.quantile(m85.ent(Pv), 0.9)
            fixed = H > tv; moving = m85.online(H, tv); both = m88.quota(H, tv); qo = quota_on(fixed)
            eq = H > np.quantile(H, 1 - both.mean())
            masks.append({"fixed": fixed, "moving": moving, "quota_only": qo, "both": both, "equal_vol": eq})
            PS.append(Pt.argmax(1))
        K = max(int(y.max()), *(int(p.max()) for p in PS + PE)) + 1
        pairs = []
        for s in range(5):
            for e in range(5):
                row = {"student_seed": s, "expert_seed": e}
                for k in POL:
                    p = np.where(masks[s][k], PE[e], PS[s])
                    row[f"f1_{k}"] = m109.mf1(y, p, K); row[f"acc_{k}"] = 100 * float((p == y).mean())
                    row[f"share_{k}"] = float(masks[s][k].mean())
                pairs.append(row)
        summ = {f"{m}_{k}": {"median": float(np.median([r[f"{m}_{k}"] for r in pairs])),
                             "min": float(min(r[f"{m}_{k}"] for r in pairs)), "max": float(max(r[f"{m}_{k}"] for r in pairs))}
                for m in ("f1", "acc", "share") for k in POL}
        for a, b in (("both", "fixed"), ("moving", "fixed"), ("quota_only", "fixed"), ("both", "moving"), ("both", "equal_vol"),
                     ("equal_vol", "fixed")):
            for m in ("f1", "acc"):
                d = [r[f"{m}_{a}"] - r[f"{m}_{b}"] for r in pairs]
                summ[f"{m}_{a}_minus_{b}"] = {"median": float(np.median(d)), "min": float(min(d)), "max": float(max(d)),
                                              "better": sum(x > 0.05 for x in d), "worse": sum(x < -0.05 for x in d)}
        res[f"{ds}_{task}"] = {"pairs": pairs, "summary": summ}
        S = summ
        print(f"{ds}_{task}: share fixed {S['share_fixed']['median']:.3f} moving {S['share_moving']['median']:.3f} quota_only "
              f"{S['share_quota_only']['median']:.3f} both {S['share_both']['median']:.3f} | F1 both-fixed "
              f"{S['f1_both_minus_fixed']['median']:+.3f} = moving-fixed {S['f1_moving_minus_fixed']['median']:+.3f}, "
              f"both-moving {S['f1_both_minus_moving']['median']:+.3f}; equal_vol-fixed {S['f1_equal_vol_minus_fixed']['median']:+.3f}, "
              f"both-equal_vol {S['f1_both_minus_equal_vol']['median']:+.3f} | ACC both-fixed {S['acc_both_minus_fixed']['median']:+.3f}, "
              f"equal_vol-fixed {S['acc_equal_vol_minus_fixed']['median']:+.3f}", flush=True)
    json.dump(res, open(OUT / "quota_decomposition.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
