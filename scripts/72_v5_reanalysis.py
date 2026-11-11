"""v5 re-analyses of the v4 score dumps (outputs_v4/regime/*.npz, unsw.npz); no training. Output: outputs_v5/reanalysis_v5.json.

A1  KS distance vs Case 2 with the attack share controlled. Per instance, 20 windows of 1,000 deployment flows:
    (m) window attack share MATCHED to the validation attack share, reference = validation scores (<= 20k);
    (b) BENIGN-only windows vs benign validation scores; (n) NATURAL windows at the deployment share (v4 reference).
    Setting level = median over models. AUC for Case 2 (median R*_1% < 0.5) over 18 settings and within IoT-23 (12),
    95% intervals from 2,000 bootstrap resamples of capture clusters. Spearman(KS, |share gap|) overall and IoT-23.
A2  Time-ordered commissioning, robustness: for every time-ordered setting (11 IoT-23 captures, real home) and
    K in {250, 500, 1000} (K <= half the benign flows): window of K consecutive benign flows starting (s) at the
    capture start (the v4 rule) or (r) at 20 random starts; threshold of Eq. (7); FPR on the benign flows AFTER the
    window; recall on all attack flows. Control (x): K benign flows drawn at random (exchangeable), FPR on the rest.
A4  Temporal dependence: two 300-flow benign windows in time order separated by a gap of 0, 1,000 or 10,000 flows
    (and random disjoint windows); Eq. (9) rejection rate. IoT-23 captures and the real home.
A5  The four score-based label-free signals of scripts/51 (score KS, conformal abstention, 1 - ATC, disagreement
    with a reference model of another family) recomputed on the v4 models; domain-classifier AUC depends on the data
    only and is taken from v3. Setting-level AUC for Case 2 with cluster intervals and within IoT-23.
A6  Uncertainty for Table IV: per setting, the median lab / Step 1 FPR with 95% bootstrap-over-models intervals,
    per-model min and max, worst 5% FPR, B3.
A7  Score orientation: R*_1% of the flipped score (alarm when s < t) per setting."""

import glob
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

O4 = Path("D:/shield_run/outputs_v4")
O5 = Path("D:/shield_run/outputs_v5"); O5.mkdir(exist_ok=True)
ALPHA, CKS = 0.01, 1.949
rng = np.random.default_rng(20261103)


def crit(n, m):
    return CKS * math.sqrt((n + m) / (n * m))


def ks(a_sorted, b):
    b = np.sort(b); v = np.concatenate([a_sorted, b])
    return float(np.max(np.abs(np.searchsorted(a_sorted, v, "right") / len(a_sorted) - np.searchsorted(b, v, "right") / len(b))))


def conformal(s, a=ALPHA):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def rstar(b, a):
    return float((a > np.sort(b)[int(math.ceil((1 - ALPHA) * len(b))) - 1]).mean())


def cluster_of(s):
    if s.startswith("ciciomt") or s == "cross_xciciot_to_xciciomt":
        return "ciciomt"
    if s.startswith("ciciot") or s == "cross_xciciomt_to_xciciot":
        return "ciciot"
    if s == "iot23_random_control_all":
        return "iot23_random"
    return "iot23_" + s.split("_")[-1]


def auc_ci(lab, x, groups, reps=2000):
    lab, x, groups = np.asarray(lab, bool), np.asarray(x, float), np.asarray(groups)
    pt = float(roc_auc_score(lab, x)); ug = np.unique(groups); bs = []
    for _ in range(reps):
        g = rng.choice(ug, len(ug)); b = np.concatenate([np.flatnonzero(groups == z) for z in g])
        if 0 < lab[b].sum() < len(b):
            bs.append(roc_auc_score(lab[b], x[b]))
    return {"auc": pt, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "n_boot_valid": len(bs),
            "n_pos": int(lab.sum()), "n": int(len(lab))}


def draw(pool, k):
    return rng.choice(pool, k, replace=len(pool) < k)


E12 = json.load(open(O4 / "exp12_v4.json"))
files = sorted(f for f in glob.glob(str(O4 / "regime" / "*.npz")) if not f.endswith("unsw.npz"))
D = {Path(f).stem: dict(np.load(f, allow_pickle=True)) for f in files}
settings = sorted(D)
case2 = {s: E12[s]["rstar_median"] < 0.5 for s in settings}
groups = [cluster_of(s) for s in settings]
io = np.array([s.startswith("iot23") for s in settings])
lab = np.array([case2[s] for s in settings])
out = {"settings": settings, "case2": case2, "seed": 20261103}

# ---------------------------------------------------------------- A1
A1 = {"per_setting": {}}
for s in settings:
    Z = D[s]; yv = Z["yv"].astype(bool); yt = Z["yt"].astype(bool); pv = float(yv.mean())
    km, kb, kn = [], [], []
    for i in range(len(Z["models"])):
        sv, st = Z["sv"][i].astype(np.float64), Z["st"][i].astype(np.float64)
        ref = np.sort(sv if len(sv) <= 20000 else rng.choice(sv, 20000, replace=False))
        refb = np.sort(sv[~yv] if (~yv).sum() <= 20000 else rng.choice(sv[~yv], 20000, replace=False))
        b, a = st[~yt], st[yt]; na = int(round(1000 * pv))
        km.append(np.mean([ks(ref, np.r_[draw(b, 1000 - na), draw(a, na)]) for _ in range(20)]))
        kb.append(np.mean([ks(refb, draw(b, 1000)) for _ in range(20)]))
        kn.append(np.mean([ks(ref, draw(st, 1000)) for _ in range(20)]))
    A1["per_setting"][s] = {"ks_matched": float(np.median(km)), "ks_benign": float(np.median(kb)), "ks_natural": float(np.median(kn)),
                            "val_share": pv, "dep_share": float(yt.mean()), "case2": case2[s], "cluster": cluster_of(s)}
    print("A1", s, {k: round(v, 3) if isinstance(v, float) else v for k, v in A1["per_setting"][s].items()}, flush=True)
for v in ("ks_matched", "ks_benign", "ks_natural"):
    x = np.array([A1["per_setting"][s][v] for s in settings])
    A1[v] = {"all": auc_ci(lab, x, groups), "iot23": auc_ci(lab[io], x[io], np.array(groups)[io])}
gap = np.array([abs(A1["per_setting"][s]["val_share"] - A1["per_setting"][s]["dep_share"]) for s in settings])
xn = np.array([A1["per_setting"][s]["ks_natural"] for s in settings])
iosh = np.array([s.startswith("iot23") and "random" not in s for s in settings])
A1["spearman_natural_ks_vs_gap"] = {"all": [float(spearmanr(xn, gap).statistic), float(spearmanr(xn, gap).pvalue)],
                                    "iot23_shift": [float(spearmanr(xn[iosh], gap[iosh]).statistic), float(spearmanr(xn[iosh], gap[iosh]).pvalue)]}
out["A1"] = A1
print("A1 summary", {v: (round(A1[v]["all"]["auc"], 3), np.round(A1[v]["all"]["ci"], 2).tolist(), round(A1[v]["iot23"]["auc"], 3),
                         np.round(A1[v]["iot23"]["ci"], 2).tolist()) for v in ("ks_matched", "ks_benign", "ks_natural")}, A1["spearman_natural_ks_vs_gap"], flush=True)

# ---------------------------------------------------------------- A2
Zu = dict(np.load(O4 / "regime" / "unsw.npz", allow_pickle=True)); days = [str(d) for d in Zu["days"]]


def a2_stream(b_time, a_all, K, n_rand=20):
    """b_time: benign scores in time order; a_all: attack scores. Returns FPR/recall for start, random starts, exchangeable."""
    n = len(b_time); res = {}
    tc = conformal(b_time[:K]); res["start"] = (float((b_time[K:] > tc).mean()), float((a_all > tc).mean()) if len(a_all) else None)
    hi = n - 2 * K
    fr, rr = [], []
    for st0 in (rng.integers(0, hi + 1, n_rand) if hi > 0 else [0]):
        tc = conformal(b_time[st0:st0 + K]); after = b_time[st0 + K:]
        fr.append(float((after > tc).mean())); rr.append(float((a_all > tc).mean()) if len(a_all) else np.nan)
    res["random"] = (fr, rr)
    fx = []
    for _ in range(n_rand):
        w = rng.choice(n, K, replace=False); m = np.ones(n, bool); m[w] = False
        fx.append(float((b_time[m] > conformal(b_time[w])).mean()))
    res["exch"] = fx
    return res


A2 = {}
tsets = [s for s in settings if s.startswith("iot23") and s != "iot23_random_control_all"]
for s in tsets + ["real_home"]:
    A2[s] = {}
    if s == "real_home":
        streams = [(np.concatenate([Zu[f"u_{d}"][i] for d in days]).astype(np.float64), Zu["sv"][i][Zu["yv"].astype(bool)].astype(np.float64))
                   for i in range(len(Zu["models"]))]
    else:
        Z = D[s]; yt = Z["yt"].astype(bool); o = np.argsort(Z["ts"], kind="stable")
        streams = [(Z["st"][i].astype(np.float64)[o][~yt[o]], Z["st"][i].astype(np.float64)[yt]) for i in range(len(Z["models"]))]
    nb = len(streams[0][0])
    for K in (250, 500, 1000):
        if K > nb // 2:
            continue
        r = [a2_stream(b, a, K) for b, a in streams]
        f_start = [x["start"][0] for x in r]; rec_start = [x["start"][1] for x in r]
        f_rand_med = [float(np.median(x["random"][0])) for x in r]; f_rand_all = [v for x in r for v in x["random"][0]]
        rec_rand = [float(np.nanmedian(x["random"][1])) for x in r]
        f_ex = [float(np.median(x["exch"])) for x in r]; f_ex_all = [v for x in r for v in x["exch"]]
        A2[s][str(K)] = {"n_benign": nb, "n_models": len(r),
                         "start_fpr_median": float(np.median(f_start)), "start_recall_median": float(np.median(rec_start)),
                         "random_fpr_median": float(np.median(f_rand_med)), "random_fpr_p95_all": float(np.quantile(f_rand_all, .95)),
                         "random_share_le_1pct": float(np.mean(np.array(f_rand_all) <= 0.01)),
                         "random_share_le_1p5pct": float(np.mean(np.array(f_rand_all) <= 0.015)),
                         "random_recall_median": float(np.median(rec_rand)),
                         "exch_fpr_median": float(np.median(f_ex)), "exch_fpr_p95_all": float(np.quantile(f_ex_all, .95)),
                         "exch_share_le_1p5pct": float(np.mean(np.array(f_ex_all) <= 0.015))}
        print("A2", s, K, {k: round(v, 4) if isinstance(v, float) else v for k, v in A2[s][str(K)].items()}, flush=True)
out["A2"] = A2

# ---------------------------------------------------------------- A4
A4 = {}
def a4(b, reps=10):
    res = {}
    for gp in (0, 1000, 10000):
        if len(b) < 600 + gp:
            continue
        rj = []
        for _ in range(reps):
            s0 = rng.integers(0, len(b) - 600 - gp + 1)
            rj.append(ks(np.sort(b[s0:s0 + 300]), b[s0 + 300 + gp:s0 + 600 + gp]) > crit(300, 300))
        res[str(gp)] = rj
    rr = []
    for _ in range(reps):
        p = rng.permutation(len(b))[:600]; rr.append(ks(np.sort(b[p[:300]]), b[p[300:]]) > crit(300, 300))
    res["random"] = rr
    return res
for name in tsets + ["real_home"]:
    agg = {}
    if name == "real_home":
        bs = [np.concatenate([Zu[f"u_{d}"][i] for d in days]).astype(np.float64) for i in range(len(Zu["models"]))]
    else:
        Z = D[name]; yt = Z["yt"].astype(bool); o = np.argsort(Z["ts"], kind="stable")
        bs = [Z["st"][i].astype(np.float64)[o][~yt[o]] for i in range(len(Z["models"]))]
    if len(bs[0]) < 600:
        continue
    for b in bs:
        for k, v in a4(b).items():
            agg.setdefault(k, []).extend(v)
    A4[name] = {k: {"reject": float(np.mean(v)), "n": len(v)} for k, v in agg.items()}
pool = {}
for name, v in A4.items():
    if name == "real_home":
        continue
    for k, x in v.items():
        pool.setdefault(k, [0, 0]); pool[k][0] += x["reject"] * x["n"]; pool[k][1] += x["n"]
A4["iot23_pooled"] = {k: {"reject": v[0] / v[1], "n": v[1]} for k, v in pool.items()}
out["A4"] = A4
print("A4", {k: A4[k] for k in ("iot23_pooled", "real_home")}, flush=True)

# ---------------------------------------------------------------- A5
def predictors(sv, yv, st, sref):
    pv = np.c_[1 - sv, sv]; pt = np.c_[1 - st, st]
    nc = 1 - pv[np.arange(len(yv)), yv]; q = np.quantile(nc, 0.95)
    set_size = ((1 - pt) <= q).sum(1)
    acc_v = float(((sv > 0.5).astype(int) == yv).mean()); cv, ct = pv.max(1), pt.max(1)
    t = np.quantile(cv, 1 - acc_v)
    return {"score_ks": ks(np.sort(sv), st), "conf_abst": float((set_size != 1).mean()), "atc": float(1 - (ct > t).mean()),
            "disagree": float(((st > 0.5) != (sref > 0.5)).mean())}


def ref_name(s, m, models):
    if s.startswith("cross"):
        return "mlp_s0" if m.startswith("xgb") else "xgb_s0"
    if "loco" in s:
        return "large_mlp_s0" if m.startswith(("xgb", "rf")) else "xgboost_s0"
    if s.startswith("iot23"):
        return "large_mlp_s0" if m == "xgboost_s0" else "xgboost_s0"
    return ("teacher_s0" if "teacher_s0" in models else models[0]) if m == "xgboost_s0" or "xgboost_s0" not in models else "xgboost_s0"


v3 = json.load(open("D:/shield_run/outputs_v3/regime_check.json"))["instances"] + json.load(open("D:/shield_run/outputs_v3/regime_loco.json"))["instances"]
dom = {s: float(np.median([r["dom_auc"] for r in v3 if r["setting"] == s])) for s in settings}
A5 = {"per_setting": {}}
for s in settings:
    Z = D[s]; models = [str(m) for m in Z["models"]]; yv = Z["yv"].astype(int)
    rows = []
    for i, m in enumerate(models):
        rn = ref_name(s, m, models); j = models.index(rn) if rn in models else (1 if i == 0 else 0)
        rows.append(predictors(Z["sv"][i].astype(np.float64), yv, Z["st"][i].astype(np.float64), Z["st"][j].astype(np.float64)))
    A5["per_setting"][s] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    A5["per_setting"][s]["dom_auc"] = dom[s]
for k in ("score_ks", "disagree", "atc", "dom_auc", "conf_abst"):
    x = np.array([A5["per_setting"][s][k] for s in settings])
    A5[k] = {"all": auc_ci(lab, x, groups), "iot23": float(roc_auc_score(lab[io], x[io]))}
out["A5"] = A5
print("A5", {k: (round(A5[k]["all"]["auc"], 3), np.round(A5[k]["all"]["ci"], 2).tolist(), round(A5[k]["iot23"], 3))
             for k in ("score_ks", "disagree", "atc", "dom_auc", "conf_abst")}, flush=True)

# ---------------------------------------------------------------- A6, A7
A6, A7 = {}, {}
def boot_med(v, reps=2000):
    v = np.asarray(v); b = [np.median(rng.choice(v, len(v))) for _ in range(reps)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
def home_per_model():
    """Real home, as scripts/70: lab FPR over days 1 to 3; Step 1 on the first 1,000 flows of day 1, FPR on the rest;
    recall = lab recall on IoT-23 validation attacks."""
    yv = Zu["yv"].astype(bool); pm = {"fpr_lab": [], "fpr_com": [], "rec_com": []}
    for i in range(len(Zu["models"])):
        sv = Zu["sv"][i].astype(np.float64); t0 = np.quantile(sv[~yv], 1 - ALPHA)
        allu = np.concatenate([Zu[f"u_{d}"][i] for d in days]).astype(np.float64)
        tc = conformal(allu[:1000])
        pm["fpr_lab"].append(float((allu > t0).mean())); pm["fpr_com"].append(float((allu[1000:] > tc).mean()))
        pm["rec_com"].append(float((sv[yv] > tc).mean()))
    return pm


for s in list(E12):
    pm = E12[s]["per_model"] if s != "real_home" else home_per_model()
    fl, fc = np.array(pm["fpr_lab"]), np.array(pm["fpr_com"])
    A6[s] = {"n_models": len(fl), "lab_median": float(np.median(fl)), "lab_ci": boot_med(fl), "lab_min": float(fl.min()), "lab_max": float(fl.max()),
             "lab_p95": float(np.quantile(fl, .95)), "com_median": float(np.median(fc)), "com_ci": boot_med(fc), "com_min": float(fc.min()),
             "com_max": float(fc.max()), "com_p95": float(np.quantile(fc, .95)), "b3_lab": float(np.mean(fl <= .03)), "b3_com": float(np.mean(fc <= .03)),
             "rec_com_median": float(np.median(pm["rec_com"])), "rec_com_ci": boot_med(pm["rec_com"]),
             "rec_com_min": float(np.min(pm["rec_com"])), "rec_com_max": float(np.max(pm["rec_com"]))}
    if s in D:
        Z = D[s]; yt = Z["yt"].astype(bool)
        rf = [rstar(-Z["st"][i][~yt].astype(np.float64), -Z["st"][i][yt].astype(np.float64)) for i in range(len(Z["models"]))]
        A7[s] = {"rstar": E12[s]["rstar_median"], "rstar_flipped": float(np.median(rf)), "auc": E12[s]["auc_median"]}
out["A6"], out["A7"] = A6, A7
print("A7", {s: (round(v["rstar"], 3), round(v["rstar_flipped"], 3)) for s, v in A7.items()}, flush=True)
json.dump(out, open(O5 / "reanalysis_v5.json", "w"), indent=1)
print("saved", O5 / "reanalysis_v5.json")
