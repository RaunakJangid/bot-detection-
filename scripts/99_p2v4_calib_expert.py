"""P2 v4, T2 + T3: calibration diagnostic and expert-value analysis under shift. CPU only, existing logits/scores.

Settings: CICIoT2023 34-class (control), CICIoMT2024 binary / 6-class / 19-class, IoT-23 capture-shift 1-1 and 35-1,
same-family 49-1 and 7-1. Students shield_s0-4 (CIC) / small_mlp_kd_s0-4 (IoT-23); experts teacher_s0 / ensemble_s0.
T2 (calibration as a diagnostic, not as the deployment rule):
  temperature T fitted per seed on the validation set (NLL); expected calibration error (15 bins, max-probability) of
  the student on validation and on each test stream, before and after scaling; escalated share with the validation
  90th-percentile threshold, before and after scaling; overlap (Jaccard) of the escalated sets before and after scaling.
T3 (expert value): macro-F1 of student, expert and cascade (validation threshold, 10% budget); per-class recall and
  benign precision of student and expert; bootstrap (1,000 resamples over flows, seed 0 student) 95% CI of the
  expert - student macro-F1 difference and of each per-class recall difference.
Output: outputs_v5/p2/calib_expert.json."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, precision_score, recall_score

from shield.data.datasets import load_processed
from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
rng = np.random.default_rng(99)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ent(P):
    P = np.clip(P, 1e-12, 1); return -(P * np.log(P)).sum(1)


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def fit_T(Z, y):
    nll = lambda t: -np.mean(np.log(np.clip(softmax(Z / t)[np.arange(len(y)), y], 1e-12, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def mf1(y, p):
    return 100 * f1_score(y, p, average="macro")


def analyse(name, Zv, yv, Zt, yt, pe, benign):
    T = fit_T(Zv, yv)
    Pv, Pt, Pvs, Pts = softmax(Zv), softmax(Zt), softmax(Zv / T), softmax(Zt / T)
    tv, tvs = np.quantile(ent(Pv), 0.9), np.quantile(ent(Pvs), 0.9)
    e_raw, e_cal = ent(Pt) > tv, ent(Pts) > tvs
    ps = Pt.argmax(1)
    labels = np.unique(yt)
    out = {"T": T, "ece_val": ece(Pv, yv), "ece_val_scaled": ece(Pvs, yv), "ece_test": ece(Pt, yt), "ece_test_scaled": ece(Pts, yt),
           "e_raw": float(e_raw.mean()), "e_scaled": float(e_cal.mean()),
           "jaccard_escalated": float((e_raw & e_cal).sum() / max((e_raw | e_cal).sum(), 1)),
           "f1_student": mf1(yt, ps), "f1_expert": mf1(yt, pe), "f1_cascade": mf1(yt, np.where(e_raw, pe, ps)),
           "recall_student": dict(zip(map(int, labels), map(float, recall_score(yt, ps, labels=labels, average=None, zero_division=0)))),
           "recall_expert": dict(zip(map(int, labels), map(float, recall_score(yt, pe, labels=labels, average=None, zero_division=0)))),
           "benign_precision_student": float(precision_score(yt == benign, ps == benign, zero_division=0)),
           "benign_precision_expert": float(precision_score(yt == benign, pe == benign, zero_division=0))}
    return out, ps


def boot(yt, ps, pe, labels):
    d, rd = [], {int(c): [] for c in labels}
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt)); y, a, b = yt[i], ps[i], pe[i]
        d.append(mf1(y, b) - mf1(y, a))
        ra = recall_score(y, a, labels=labels, average=None, zero_division=0)
        rb = recall_score(y, b, labels=labels, average=None, zero_division=0)
        for c, x, z in zip(labels, ra, rb):
            rd[int(c)].append(z - x)
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return {"f1_expert_minus_student_ci": q(d), "recall_diff_ci": {c: q(v) for c, v in rd.items()}}


def main():
    P = Paths(False); res = {}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        yv, yt = d.splits["validation"].y, d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"
        pe = np.load(lc / "teacher_s0_test.npy").argmax(1)
        benign = int(np.bincount(yt).argmax()) if task != "binary" else 0
        names = list(getattr(d, "class_names", []) or [])
        if names:
            bn = [i for i, n in enumerate(names) if "enign" in str(n)]
            benign = bn[0] if bn else benign
        seeds = []
        for s in range(5):
            Zv = np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64); Zt = np.load(lc / f"shield_s{s}_test.npy").astype(np.float64)
            o, ps = analyse(f"{ds}_{task}", Zv, yv, Zt, yt, pe, benign)
            if s == 0:
                o["bootstrap"] = boot(yt, ps, pe, np.unique(yt))
            seeds.append(o)
        res[f"{ds}_{task}"] = {"benign_class": benign, "seeds": seeds}
        print(ds, task, {k: round(float(np.median([x[k] for x in seeds])), 4) for k in
                         ("T", "ece_val", "ece_test", "ece_test_scaled", "e_raw", "e_scaled", "jaccard_escalated",
                          "f1_student", "f1_expert", "f1_cascade", "benign_precision_student", "benign_precision_expert")},
              "F1 diff CI", seeds[0]["bootstrap"]["f1_expert_minus_student_ci"], flush=True)
    for design in ("capture_shift", "same_family"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        for cap in ("CTU-IoT-Malware-Capture-1-1", "CTU-IoT-Malware-Capture-35-1", "CTU-IoT-Malware-Capture-49-1",
                    "CTU-IoT-Malware-Capture-7-1"):
            m = Z["capture"] == cap
            if not m.any():
                continue
            yt = Z["yt"][m].astype(int); yv = Z["yv"].astype(int)
            pe = (Z["st"][models.index("ensemble_s0")][m] > 0.5).astype(int)
            seeds = []
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
                Zv = np.c_[np.zeros(len(yv)), lg(Z["sv"][i])]; Zt = np.c_[np.zeros(m.sum()), lg(Z["st"][i][m])]
                o, ps = analyse(cap, Zv, yv, Zt, yt, pe, 0)
                if s == 0:
                    o["bootstrap"] = boot(yt, ps, pe, np.unique(yt))
                seeds.append(o)
            res[f"iot23_{design}_{cap}"] = {"benign_class": 0, "seeds": seeds}
            print(design, cap, {k: round(float(np.median([x[k] for x in seeds])), 4) for k in
                                ("T", "ece_val", "ece_test", "e_raw", "e_scaled", "jaccard_escalated", "f1_student",
                                 "f1_expert", "f1_cascade", "benign_precision_student", "benign_precision_expert")},
                  "F1 diff CI", seeds[0]["bootstrap"]["f1_expert_minus_student_ci"], flush=True)
    json.dump(res, open(OUT / "calib_expert.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
