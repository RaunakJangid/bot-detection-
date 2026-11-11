"""v4 Experiments 1 and 2 from the dumped scores (outputs_v4/regime/*.npz, unsw.npz); no training.

Per setting and model: lab threshold t0 = 99th percentile of validation benign scores (Eq. 1, alpha = 1%); realised FPR
and recall on the deployment flows; commissioned threshold t_c = split-conformal 1% quantile of K = 1,000 deployment
benign flows (Eq. 8): the first K benign flows in time order where flow times exist (IoT-23; K = half the benign flows
if fewer than 2,000), else 20 random draws (CICIoMT2024, CICIoT2023, cross-dataset); FPR and recall on the remaining
deployment flows. Also R*_1% (oracle recall at <= 1% deployment FPR) and the case (Case 2 if median R* < 0.5).
Real home: lab FPR over days 1 to 3; t_c on the first 1,000 flows of day 1, FPR on the rest; recall = lab recall on
IoT-23 validation attacks (the traces contain no attacks). Output: outputs_v4/exp12_v4.json."""

import glob
import math
from pathlib import Path

import numpy as np

from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("v4_exp12")
A = 0.01


def conformal(s, a=A):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def rstar(b, a):
    t = np.sort(b)[int(math.ceil((1 - A) * len(b))) - 1]
    return float((a > t).mean())


def summ(rows):
    f0 = np.array([r["fpr_lab"] for r in rows]); fc = np.array([r["fpr_com"] for r in rows])
    return {"n_models": len(rows), "fpr_lab_median": float(np.median(f0)), "p95_r_lab": float(np.quantile(f0, 0.95) / A),
            "b3_lab": float(np.mean(f0 / A <= 3)), "fpr_com_median": float(np.median(fc)),
            "p95_r_com": float(np.quantile(fc, 0.95) / A), "b3_com": float(np.mean(fc / A <= 3)),
            "recall_lab_median": float(np.median([r["rec_lab"] for r in rows])),
            "recall_com_median": float(np.median([r["rec_com"] for r in rows])),
            "rstar_median": float(np.median([r["rstar"] for r in rows])) if rows[0].get("rstar") is not None else None,
            "auc_median": float(np.median([r["auc"] for r in rows])) if rows[0].get("auc") is not None else None,
            "k_used": rows[0]["k"], "time_ordered": rows[0]["timed"]}


def main():
    rng = np.random.default_rng(0)
    out = {}
    for f in sorted(glob.glob("D:/shield_run/outputs_v4/regime/*.npz")):
        s = Path(f).stem
        if s == "unsw":
            continue
        Z = np.load(f, allow_pickle=True); yv = Z["yv"].astype(bool); yt = Z["yt"].astype(bool)
        timed = bool(np.isfinite(Z["ts"]).all()) and s != "iot23_random_control_all"
        rows = []
        for i in range(len(Z["models"])):
            sv, st = Z["sv"][i].astype(np.float64), Z["st"][i].astype(np.float64)
            t0 = np.quantile(sv[~yv], 1 - A)
            b, a = st[~yt], st[yt]
            r = {"fpr_lab": float((b > t0).mean()), "rec_lab": float((a > t0).mean()), "rstar": rstar(b, a),
                 "auc": float(Z["target_auc"][i]), "timed": timed}
            if timed:
                o = np.argsort(Z["ts"], kind="stable"); ob = o[~yt[o]]
                K = 1000 if len(ob) >= 2000 else len(ob) // 2
                tc = conformal(st[ob[:K]])
                rest_b = st[ob[K:]]
                r.update(fpr_com=float((rest_b > tc).mean()), rec_com=float((a > tc).mean()), k=K)
            else:
                fp, rc = [], []
                for _ in range(20):
                    w = rng.choice(len(b), 1000, replace=False); m = np.ones(len(b), bool); m[w] = False
                    tc = conformal(b[w]); fp.append(float((b[m] > tc).mean())); rc.append(float((a > tc).mean()))
                r.update(fpr_com=float(np.median(fp)), rec_com=float(np.median(rc)), k=1000)
            rows.append(r)
        out[s] = summ(rows)
        out[s]["per_model"] = {"fpr_lab": [r["fpr_lab"] for r in rows], "fpr_com": [r["fpr_com"] for r in rows],
                               "rec_lab": [r["rec_lab"] for r in rows], "rec_com": [r["rec_com"] for r in rows]}
        out[s]["case2"] = out[s]["rstar_median"] < 0.5
        v = out[s]
        log.info("%-48s n=%2d K=%4d | lab FPR %5.2f%% p95r %5.2f B3 %3.0f%% rec %5.1f | com FPR %5.2f%% p95r %5.2f B3 %3.0f%% rec %5.1f | R* %.2f AUC %.3f",
                 s, v["n_models"], v["k_used"], 100 * v["fpr_lab_median"], v["p95_r_lab"], 100 * v["b3_lab"],
                 100 * v["recall_lab_median"], 100 * v["fpr_com_median"], v["p95_r_com"], 100 * v["b3_com"],
                 100 * v["recall_com_median"], v["rstar_median"], v["auc_median"])
    Zu = np.load("D:/shield_run/outputs_v4/regime/unsw.npz", allow_pickle=True)
    days = [str(d) for d in Zu["days"]]; yv = Zu["yv"].astype(bool); rows = []
    for i in range(len(Zu["models"])):
        sv = Zu["sv"][i].astype(np.float64); t0 = np.quantile(sv[~yv], 1 - A)
        allu = np.concatenate([Zu[f"u_{d}"][i] for d in days]).astype(np.float64)
        d1 = Zu[f"u_{days[0]}"][i].astype(np.float64); tc = conformal(d1[:1000])
        rest = np.concatenate([d1[1000:]] + [Zu[f"u_{d}"][i] for d in days[1:]]).astype(np.float64)
        rows.append({"fpr_lab": float((allu > t0).mean()), "rec_lab": float((sv[yv] > t0).mean()),
                     "fpr_com": float((rest > tc).mean()), "rec_com": float((sv[yv] > tc).mean()),
                     "rstar": None, "auc": None, "k": 1000, "timed": True})
    out["real_home"] = summ(rows)
    out["real_home"]["fpr_lab_range"] = [float(min(r["fpr_lab"] for r in rows)), float(max(r["fpr_lab"] for r in rows))]
    out["real_home"]["fpr_com_range"] = [float(min(r["fpr_com"] for r in rows)), float(max(r["fpr_com"] for r in rows))]
    log.info("real_home %s", out["real_home"])
    save_json(out, "D:/shield_run/outputs_v4/exp12_v4.json")


if __name__ == "__main__":
    main()
