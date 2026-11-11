"""P2 v4: effect of the escalation policy on detection quality across ALL student x expert seeds, in accuracy AND
macro-F1 (reconciliation check: scripts/104 reported, for CICIoMT 19-class, a +2.38 point gain from quota rejections
measured in ACCURACY with expert seed 0 only, while scripts/107 found the expert ahead in macro-F1 in 16/25 seed pairs).

Per setting and student x expert pair, predictions on the same flows under five policies:
  student      student only
  fixed        escalate every candidate (entropy > validation 90th percentile); expert answers those
  quota        deployed rule (online quantile + quota, scripts/88 quota()); expert answers admitted flows
  expert       expert only
and the derived quantity quota - fixed (= effect of the rejections). Metrics: accuracy and macro-F1 (bincount confusion,
classes present in truth or prediction), same flow population and denominators for every policy.
CIC: CICIoT2023 34-class, CICIoMT2024 binary / 6-class / 19-class (flow order: rng(0) permutation, as scripts/104),
5 x 5 pairs. IoT-23: 1-1, 35-1, 49-1, 7-1 in time order, 5 students x 3 ensembles.
Output: outputs_v5/p2/quota_effect_seeds.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m85, m88 = _l("85_p2v4_cap_quality.py"), _l("88_p2v4_escalation_attack.py")
POL = ("student", "fixed", "quota", "expert")


def mf1(y, p, K):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; present = (cm.sum(1) + cm.sum(0)) > 0
    return 100 * float(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)[present].mean())


def evaluate(y, PS, PE, HS, TV, K):
    adm = [m88.quota(HS[s], TV[s]) for s in range(len(PS))]; cand = [HS[s] > TV[s] for s in range(len(PS))]
    pairs = []
    for s in range(len(PS)):
        for e in range(len(PE)):
            pr = {"student": PS[s], "fixed": np.where(cand[s], PE[e], PS[s]), "quota": np.where(adm[s], PE[e], PS[s]), "expert": PE[e]}
            row = {"student_seed": s, "expert_seed": e, "cand_share": float(cand[s].mean()), "adm_share": float(adm[s].mean())}
            for k, p in pr.items():
                row[f"acc_{k}"] = 100 * float((p == y).mean()); row[f"f1_{k}"] = mf1(y, p, K)
            row["acc_quota_minus_fixed"] = row["acc_quota"] - row["acc_fixed"]; row["f1_quota_minus_fixed"] = row["f1_quota"] - row["f1_fixed"]
            pairs.append(row)
    summ = {}
    for key in [f"{m}_{k}" for m in ("acc", "f1") for k in POL] + ["acc_quota_minus_fixed", "f1_quota_minus_fixed"]:
        v = [r[key] for r in pairs]; summ[key] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
    for m in ("acc", "f1"):
        d = [r[f"{m}_quota_minus_fixed"] for r in pairs]
        summ[f"{m}_quota_vs_fixed_counts"] = {"quota_better": sum(x > 0.05 for x in d), "fixed_better": sum(x < -0.05 for x in d),
                                              "tie": sum(abs(x) <= 0.05 for x in d)}
    return {"pairs": pairs, "summary": summ}


def report(name, r):
    s = r["summary"]
    print(name, " | ".join(f"{m} " + " ".join(f"{k} {s[f'{m}_{k}']['median']:.2f}" for k in POL) for m in ("acc", "f1")),
          f"| quota-fixed acc {s['acc_quota_minus_fixed']['median']:+.2f} [{s['acc_quota_minus_fixed']['min']:+.2f},{s['acc_quota_minus_fixed']['max']:+.2f}]"
          f" {s['acc_quota_vs_fixed_counts']} f1 {s['f1_quota_minus_fixed']['median']:+.2f} [{s['f1_quota_minus_fixed']['min']:+.2f},"
          f"{s['f1_quota_minus_fixed']['max']:+.2f}] {s['f1_quota_vs_fixed_counts']}", flush=True)


def main():
    from shield.data.datasets import load_processed
    from shield.eval.metrics import softmax
    from shield.utils.config import Paths
    P = Paths(False); res = {}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        yt = load_processed(P.processed(ds), task, splits=("test",)).splits["test"].y.astype(np.int64)
        lc = P.v3 / "logits" / f"{ds}_{task}"; o = np.random.default_rng(0).permutation(len(yt)); y = yt[o]
        PS, HS, TV = [], [], []
        for s in range(5):
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))[o]
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            PS.append(Pt.argmax(1)); HS.append(m85.ent(Pt)); TV.append(np.quantile(m85.ent(Pv), 0.9))
        PE = [np.load(lc / f"teacher_s{t}_test.npy").argmax(1)[o] for t in range(5)]
        K = max(int(y.max()), *(int(p.max()) for p in PS + PE)) + 1
        res[f"{ds}_{task}"] = r = evaluate(y, PS, PE, HS, TV, K); report(f"{ds}_{task}", r)
    for design, cap in (("capture_shift", "CTU-IoT-Malware-Capture-1-1"), ("capture_shift", "CTU-IoT-Malware-Capture-35-1"),
                        ("same_family", "CTU-IoT-Malware-Capture-49-1"), ("same_family", "CTU-IoT-Malware-Capture-7-1")):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        m = Z["capture"] == cap; o = np.argsort(Z["ts"][m], kind="stable"); y = Z["yt"][m][o].astype(np.int64)
        st = lambda n: Z["st"][models.index(n)][m][o]
        PS, HS, TV = [], [], []
        for s in range(5):
            q = st(f"small_mlp_kd_s{s}"); sv = Z["sv"][models.index(f"small_mlp_kd_s{s}")]
            PS.append((q > 0.5).astype(np.int64)); HS.append(m85.ent(np.c_[1 - q, q])); TV.append(np.quantile(m85.ent(np.c_[1 - sv, sv]), 0.9))
        PE = [((st(f"xgboost_s{j}") + st(f"large_mlp_s{j}")) / 2 > 0.5).astype(np.int64) for j in range(3)]
        res[cap] = r = evaluate(y, PS, PE, HS, TV, 2); report(cap[-4:], r)
    json.dump(res, open(OUT / "quota_effect_seeds.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
