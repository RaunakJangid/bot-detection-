"""v4 regime analyses, step 2 (reads outputs_v4/regime/*.npz from scripts/66 and 68; no training).

A  Case definitions. v3: setting is Case 2 if the median deployment ROC-AUC < 0.9. v4: utility at the operating point,
   R*_a = max TPR over thresholds with deployment FPR <= a (a = 1%, oracle, labels used for evaluation only); Case 2 if
   R*_a < rho. Cutoff sweeps for both; KS-distance AUC for flagging Case 2 at setting level (median over models).
B  Uncertainty. Bootstrap over settings (as v3) vs cluster bootstrap over shared data: settings that share deployment
   captures form one cluster (the three CICIoMT2024 tasks and the CICIoT->CICIoMT transfer; each IoT-23 capture across
   the same-family / capture-shift / LOCO designs; CICIoT2023 control with the CICIoMT->CICIoT transfer).
C  The deployable rule. Eq. (10) (two-sample KS at alpha = 0.001) on the full samples and on fixed windows of m = 300
   and 1,000 deployment flows (contiguous in time where flow times exist, random otherwise; reference = validation
   scores, at most 20k); warning share on Case 1 vs Case 2. Then a threshold tau on D_KS chosen by Youden's J on all
   clusters but one and applied to the held-out cluster (leave-one-cluster-out): sensitivity, specificity (Wilson CI).
D  Attack share. Deployment windows of 1,000 flows resampled at attack share 0, 1, 5, 10, 20, 50%: D_KS, its Case-2
   AUC at each share, and the Eq. (10) / tau warning share on Case 1 settings.
E  Temporal dependence (null, no shift by construction). Two disjoint windows of m = 300 benign flows from the same
   stream: contiguous in time vs random. Eq. (10) rejection rate (nominal 0.001). Streams: benign flows of each IoT-23
   deployment capture (time-ordered) and the real home: first 1,000 flows of day 1 vs later 300-flow windows (the live
   design), vs random disjoint subsets.
F  Commissioning contamination. K = 1,000 window with 0, 1, 5, 10% attack flows; threshold of Eq. (8); realised FPR on
   the remaining benign flows, recall on the remaining attacks, threshold movement, spread over windows. Case 1
   settings: CICIoMT2024 (random windows, 20 draws), IoT-23 captures with ROC-AUC >= 0.9 and enough benign flows (first
   benign flows in time, attacks injected at random, 20 draws), real home (first 1,000 flows of day 1, IoT-23
   validation attacks injected, FPR over days 1-3, lab recall on the other validation attacks)."""

import glob
import math
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("regime_v4")
ALPHA = 0.01
C_KS = 1.949                                   # two-sample KS critical constant at significance 0.001


def crit(n, m):
    return C_KS * math.sqrt((n + m) / (n * m))


def ks_sorted(a_sorted, b):
    b = np.sort(b)
    v = np.concatenate([a_sorted, b])
    return float(np.max(np.abs(np.searchsorted(a_sorted, v, "right") / len(a_sorted)
                               - np.searchsorted(b, v, "right") / len(b))))


def conformal(s, a=ALPHA):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def rstar(b, a, alpha=ALPHA):
    t = np.sort(b)[int(math.ceil((1 - alpha) * len(b))) - 1]
    return float((a > t).mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def cluster_of(s):
    if s.startswith("ciciomt") or s == "cross_xciciot_to_xciciomt":
        return "ciciomt"
    if s.startswith("ciciot") or s == "cross_xciciomt_to_xciciot":
        return "ciciot"
    if s == "iot23_random_control_all":
        return "iot23_random"
    return "iot23_" + s.split("_")[-1]


def auc_boot(lab, score, groups, rng, reps=2000):
    lab, score, groups = np.asarray(lab), np.asarray(score), np.asarray(groups)
    point = float(roc_auc_score(lab, score)) if 0 < lab.sum() < len(lab) else None
    naive, clus = [], []
    ug = np.unique(groups)
    for _ in range(reps):
        b = rng.integers(0, len(lab), len(lab))
        if 0 < lab[b].sum() < len(b):
            naive.append(roc_auc_score(lab[b], score[b]))
        g = rng.choice(ug, len(ug)); b = np.concatenate([np.flatnonzero(groups == x) for x in g])
        if 0 < lab[b].sum() < len(b):
            clus.append(roc_auc_score(lab[b], score[b]))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [None, None]
    return {"auc": point, "ci_naive": q(naive), "ci_cluster": q(clus), "n_pos": int(lab.sum()), "n": int(len(lab))}


def youden_tau(x, lab):
    xs = np.unique(x)
    cand = np.r_[xs[0] - 1e-9, (xs[1:] + xs[:-1]) / 2, xs[-1] + 1e-9]
    best, bt = -2, cand[0]
    for t in cand:
        p = x > t
        j = (p & lab).sum() / max(lab.sum(), 1) - (p & ~lab).sum() / max((~lab).sum(), 1)
        if j > best:
            best, bt = j, t
    return float(bt)


def windows(n, m, ts, k, rng):
    """k index windows of size m over n flows: contiguous in time order if ts is finite, else random."""
    if np.isfinite(ts).all():
        o = np.argsort(ts, kind="stable")
        if n <= m:
            return [o]
        st = rng.integers(0, n - m + 1, k)
        return [o[s:s + m] for s in st]
    return [rng.choice(n, min(m, n), replace=False) for _ in range(k)]


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    d4 = Path(str(P.v3).replace("outputs_v3", "outputs_v4")) / "regime"
    rng = np.random.default_rng(0)
    files = sorted(f for f in glob.glob(str(d4 / "*.npz")) if not f.endswith("unsw.npz"))
    D = {Path(f).stem: dict(np.load(f, allow_pickle=True)) for f in files}
    settings = sorted(D)
    log.info("settings: %d (%s)", len(settings), ", ".join(settings))
    inst = []
    for s in settings:
        Z = D[s]; yt = Z["yt"].astype(bool); yv = Z["yv"]
        for i, m in enumerate(Z["models"]):
            sv, st = Z["sv"][i].astype(np.float64), Z["st"][i].astype(np.float64)
            dks = float(ks_2samp(sv, st).statistic)
            inst.append({"setting": s, "cluster": cluster_of(s), "model": str(m), "i": i,
                         "auc": float(Z["target_auc"][i]), "rstar": rstar(st[~yt], st[yt]), "ks": dks,
                         "eq10_full": bool(dks > crit(len(sv), len(st))), "nv": len(sv), "nt": len(st)})
    out = {"n_settings": len(settings), "n_instances": len(inst), "settings": {}}
    med = lambda s, k: float(np.median([r[k] for r in inst if r["setting"] == s]))
    S = {s: {"cluster": cluster_of(s), "auc": med(s, "auc"), "rstar": med(s, "rstar"), "ks": med(s, "ks"),
             "n_models": sum(r["setting"] == s for r in inst)} for s in settings}
    out["settings"] = S
    for s in settings:
        log.info("%-48s %-14s AUC %.3f  R* %.3f  KS %.3f", s, S[s]["cluster"], S[s]["auc"], S[s]["rstar"], S[s]["ks"])
    groups = [S[s]["cluster"] for s in settings]
    ks_set = np.array([S[s]["ks"] for s in settings])
    auc_set = np.array([S[s]["auc"] for s in settings]); r_set = np.array([S[s]["rstar"] for s in settings])

    # ---- A + B: case definitions, KS AUC with naive vs cluster bootstrap
    A = {}
    for c in (0.80, 0.85, 0.90, 0.95):
        A[f"auc<{c}"] = auc_boot(auc_set < c, ks_set, groups, rng)
    for rho in (0.5, 0.7, 0.8, 0.9):
        A[f"R*<{rho}"] = auc_boot(r_set < rho, ks_set, groups, rng)
    A["label_disagreement_auc0.9_vs_R*0.5"] = [s for s in settings if (S[s]["auc"] < 0.9) != (S[s]["rstar"] < 0.5)]
    A["label_disagreement_auc0.9_vs_R*0.8"] = [s for s in settings if (S[s]["auc"] < 0.9) != (S[s]["rstar"] < 0.8)]
    ia = np.array([r["auc"] for r in inst]); ir = np.array([r["rstar"] for r in inst]); ik = np.array([r["ks"] for r in inst])
    A["instance_level"] = {"auc<0.9": float(roc_auc_score(ia < 0.9, ik)), "R*<0.5": float(roc_auc_score(ir < 0.5, ik)),
                           "R*<0.8": float(roc_auc_score(ir < 0.8, ik))}
    A["n_clusters"] = len(set(groups))
    out["A_case_definitions"] = A
    for k, v in A.items():
        log.info("A %-36s %s", k, v)

    # ---- C: Eq. (10) full sample and fixed windows; leave-one-cluster-out tau
    C = {"eq10_full": {}}
    for lab_name, lab_fn in (("auc<0.9", lambda r: r["auc"] < 0.9), ("R*<0.5", lambda r: r["rstar"] < 0.5)):
        c1 = [r["eq10_full"] for r in inst if not lab_fn(r)]; c2 = [r["eq10_full"] for r in inst if lab_fn(r)]
        C["eq10_full"][lab_name] = {"warn_case1": float(np.mean(c1)), "n_case1": len(c1),
                                    "warn_case2": float(np.mean(c2)), "n_case2": len(c2)}
    win = {}
    for m in (300, 1000):
        for r in inst:
            Z = D[r["setting"]]; sv = Z["sv"][r["i"]].astype(np.float64)
            ref = np.sort(sv if len(sv) <= 20_000 else rng.choice(sv, 20_000, replace=False))
            st = Z["st"][r["i"]].astype(np.float64)
            ds = [ks_sorted(ref, st[w]) for w in windows(len(st), m, Z["ts"], 20, rng)]
            mm = min(m, len(st))
            r[f"win{m}_warn"] = float(np.mean([d > crit(len(ref), mm) for d in ds]))
            r[f"win{m}_ks"] = float(np.mean(ds))
        for lab_name, lab_fn in (("auc<0.9", lambda r: r["auc"] < 0.9), ("R*<0.5", lambda r: r["rstar"] < 0.5)):
            win[f"m{m}_{lab_name}"] = {"warn_case1": float(np.mean([r[f"win{m}_warn"] for r in inst if not lab_fn(r)])),
                                       "warn_case2": float(np.mean([r[f"win{m}_warn"] for r in inst if lab_fn(r)]))}
    C["eq10_windows"] = win
    ks1000_set = np.array([med(s, "win1000_ks") for s in settings])
    C["auc_window1000_ks"] = {"auc<0.9": auc_boot(auc_set < 0.9, ks1000_set, groups, rng),
                              "R*<0.5": auc_boot(r_set < 0.5, ks1000_set, groups, rng)}
    # commissioning reference (the live gateway's reference): first K benign deployment flows (time order if known,
    # else a random benign sample), K = min(1000, half the benign flows); 20 windows of 1,000 later flows
    for r in inst:
        Z = D[r["setting"]]; yt = Z["yt"].astype(bool); st = Z["st"][r["i"]].astype(np.float64)
        timed = np.isfinite(Z["ts"]).all()
        o = np.argsort(Z["ts"], kind="stable") if timed else rng.permutation(len(st))
        ob = o[~yt[o]]; K = min(1000, len(ob) // 2)
        ref = np.sort(st[ob[:K]]); rest = np.setdiff1d(o, ob[:K], assume_unique=True)
        rest = rest[np.argsort(Z["ts"][rest], kind="stable")] if timed else rest
        mm = min(1000, len(rest))
        ds = [ks_sorted(ref, st[rest[w]]) for w in windows(len(rest), mm, Z["ts"][rest] if timed else np.full(len(rest), np.nan), 20, rng)]
        r["com_ks"] = float(np.mean(ds)); r["com_warn"] = float(np.mean([d > crit(K, mm) for d in ds]))
    com_set = np.array([med(s, "com_ks") for s in settings])
    C["commissioning_reference"] = {
        "auc_R*<0.5": auc_boot(r_set < 0.5, com_set, groups, rng), "auc_auc<0.9": auc_boot(auc_set < 0.9, com_set, groups, rng),
        "eq10_warn_case1": float(np.mean([r["com_warn"] for r in inst if r["rstar"] >= 0.5])),
        "eq10_warn_case2": float(np.mean([r["com_warn"] for r in inst if r["rstar"] < 0.5]))}
    loco = {}
    for lab_name, labs in (("auc<0.9", auc_set < 0.9), ("R*<0.8", r_set < 0.8), ("R*<0.5", r_set < 0.5)):
        for feat_name, x in (("ks_full", ks_set), ("ks_win1000", ks1000_set), ("ks_commissioning", com_set)):
            pred = np.zeros(len(settings), bool); taus = []
            for g in set(groups):
                te = np.array([gg == g for gg in groups])
                if labs[~te].sum() == 0 or labs[~te].sum() == (~te).sum():
                    pred[te] = False; continue
                t = youden_tau(x[~te], labs[~te]); taus.append(t)
                pred[te] = x[te] > t
            tp, fn = int((pred & labs).sum()), int((~pred & labs).sum())
            tn, fp = int((~pred & ~labs).sum()), int((pred & ~labs).sum())
            loco[f"{lab_name}|{feat_name}"] = {"sens": tp / max(tp + fn, 1), "sens_ci": wilson(tp, tp + fn),
                                               "spec": tn / max(tn + fp, 1), "spec_ci": wilson(tn, tn + fp),
                                               "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                                               "tau_range": [float(min(taus)), float(max(taus))] if taus else None,
                                               "missed": [s for s, p_, l in zip(settings, pred, labs) if l and not p_],
                                               "false_warn": [s for s, p_, l in zip(settings, pred, labs) if p_ and not l]}
    C["loco_cluster_threshold"] = loco
    tau_all = youden_tau(ks1000_set, r_set < 0.5)                       # in-sample (all settings), for D only
    C["tau_window1000_in_sample"] = tau_all
    C["window1000_warn_at_tau_in_sample"] = {
        "case1": float(np.mean([r["win1000_ks"] > tau_all for r in inst if r["rstar"] >= 0.5])),
        "case2": float(np.mean([r["win1000_ks"] > tau_all for r in inst if r["rstar"] < 0.5]))}
    out["C_warning_rule"] = C
    for k, v in C.items():
        log.info("C %-28s %s", k, v)

    # ---- D: attack-share sweep (windows of 1,000, 5 draws)
    Dd = {}
    for pi in (0.0, 0.01, 0.05, 0.10, 0.20, 0.50):
        for r in inst:
            Z = D[r["setting"]]; yt = Z["yt"].astype(bool); st = Z["st"][r["i"]].astype(np.float64)
            sv = Z["sv"][r["i"]].astype(np.float64)
            ref = np.sort(sv if len(sv) <= 20_000 else rng.choice(sv, 20_000, replace=False))
            b, a = st[~yt], st[yt]; na = int(round(1000 * pi))
            ds = [ks_sorted(ref, np.r_[rng.choice(b, 1000 - na), rng.choice(a, na) if na else []]) for _ in range(5)]
            r[f"pi{pi}"] = float(np.mean(ds))
        x = np.array([med(s, f"pi{pi}") for s in settings])
        c1 = [r[f"pi{pi}"] for r in inst if r["rstar"] >= 0.5]
        Dd[str(pi)] = {"auc_case2_auc<0.9": auc_boot(auc_set < 0.9, x, groups, rng, 500),
                       "auc_case2_R*<0.5": auc_boot(r_set < 0.5, x, groups, rng, 500),
                       "median_ks_case1_instances": float(np.median(c1)),
                       "median_ks_case2_instances": float(np.median([r[f"pi{pi}"] for r in inst if r["rstar"] < 0.5])),
                       "eq10_warn_case1": float(np.mean([v > crit(20_000, 1000) for v in c1])),
                       "tau_warn_case1": float(np.mean([v > tau_all for v in c1])),
                       "median_ks_control_settings": float(np.median([r[f"pi{pi}"] for r in inst if r["cluster"] in ("ciciot", "iot23_random") and r["setting"] in ("ciciot_control", "iot23_random_control_all")]))}
        log.info("D share %.2f: %s", pi, {k: v if not isinstance(v, dict) else round(v["auc"] or -1, 3) for k, v in Dd[str(pi)].items()})
    out["D_attack_share"] = Dd

    # ---- E: temporal dependence null
    E = {"iot23": {}}
    rej_c, rej_r = [], []
    for s in settings:
        Z = D[s]
        if not s.startswith("iot23") or s == "iot23_random_control_all" or not np.isfinite(Z["ts"]).all():
            continue
        yt = Z["yt"].astype(bool); o = np.argsort(Z["ts"], kind="stable"); ob = o[~yt[o]]
        if len(ob) < 600:
            continue
        for i in range(len(Z["models"])):
            b = Z["st"][i].astype(np.float64)[ob]
            for _ in range(10):
                s0 = rng.integers(0, len(b) - 600 + 1)
                rej_c.append(ks_sorted(np.sort(b[s0:s0 + 300]), b[s0 + 300:s0 + 600]) > crit(300, 300))
                p = rng.permutation(len(b))[:600]
                rej_r.append(ks_sorted(np.sort(b[p[:300]]), b[p[300:]]) > crit(300, 300))
    E["iot23"] = {"contiguous_reject": float(np.mean(rej_c)), "random_reject": float(np.mean(rej_r)), "n": len(rej_c)}
    U = d4 / "unsw.npz"
    if U.exists():
        Zu = dict(np.load(U, allow_pickle=True)); days = [str(d) for d in Zu["days"]]
        live, rnd = [], []
        for i in range(len(Zu["models"])):
            com = np.sort(Zu[f"u_{days[0]}"][i][:1000].astype(np.float64))
            rest = np.concatenate([Zu[f"u_{days[0]}"][i][1000:]] + [Zu[f"u_{d}"][i] for d in days[1:]]).astype(np.float64)
            for s0 in range(0, len(rest) - 300, 1500):                        # every 1,500th flow: a 300-flow window
                live.append(ks_sorted(com, rest[s0:s0 + 300]) > crit(1000, 300))
            pool = np.concatenate([Zu[f"u_{d}"][i] for d in days]).astype(np.float64)
            for _ in range(len(live) // len(Zu["models"]) + 1):
                p = rng.permutation(len(pool))[:1300]
                rnd.append(ks_sorted(np.sort(pool[p[:1000]]), pool[p[1000:]]) > crit(1000, 300))
        t0 = Zu[f"ts_{days[0]}"]
        E["real_home"] = {"live_design_warn_share": float(np.mean(live)), "random_disjoint_warn_share": float(np.mean(rnd)),
                          "n_windows": len(live), "minutes_to_collect_first_1000_flows": float((t0[999] - t0[0]) / 60)}
    out["E_temporal_dependence"] = E
    log.info("E %s", E)

    # ---- F: commissioning contamination
    F = {}

    def contam(st_b, st_a, ordered, draws=20):
        res = {}
        for c in (0.0, 0.01, 0.05, 0.10):
            na = int(round(1000 * c)); nb = 1000 - na
            fpr, rec, thr = [], [], []
            for _ in range(draws):
                bi = np.arange(nb) if ordered else rng.choice(len(st_b), nb, replace=False)
                ai = rng.choice(len(st_a), na, replace=False) if na else np.array([], int)
                t = conformal(np.r_[st_b[bi], st_a[ai]])
                mb = np.ones(len(st_b), bool); mb[bi] = False; ma = np.ones(len(st_a), bool); ma[ai] = False
                fpr.append(float((st_b[mb] > t).mean())); rec.append(float((st_a[ma] > t).mean())); thr.append(float(t))
            res[str(c)] = (fpr, rec, thr)
        return res

    groups_F = {"ciciomt": [s for s in settings if s.startswith("ciciomt")],
                "iot23_case1": [s for s in settings if s.startswith("iot23") and s != "iot23_random_control_all"
                                and S[s]["rstar"] >= 0.5]}
    for gname, sl in groups_F.items():
        per = {str(c): {"fpr": [], "rec": [], "dthr": [], "fpr_iqr": []} for c in (0.0, 0.01, 0.05, 0.10)}
        used = []
        for s in sl:
            Z = D[s]; yt = Z["yt"].astype(bool); ordered = gname != "ciciomt"
            o = np.argsort(Z["ts"], kind="stable") if ordered else np.arange(len(yt))
            if (~yt).sum() < 1500 or yt.sum() < 200:
                continue
            used.append(s)
            for i in range(len(Z["models"])):
                st = Z["st"][i].astype(np.float64)[o]; yo = yt[o]
                r = contam(st[~yo], st[yo], ordered)
                t0 = np.median(r["0.0"][2])
                for c, (fp, rc, th) in r.items():
                    per[c]["fpr"].append(np.median(fp)); per[c]["rec"].append(np.median(rc))
                    per[c]["dthr"].append(np.median(th) - t0); per[c]["fpr_iqr"].append(np.quantile(fp, 0.75) - np.quantile(fp, 0.25))
        F[gname] = {"settings": used, **{c: {"median_fpr": float(np.median(v["fpr"])), "p95_ratio": float(np.quantile(v["fpr"], 0.95) / ALPHA),
                                              "median_recall": float(np.median(v["rec"])), "p05_recall": float(np.quantile(v["rec"], 0.05)),
                                              "median_threshold_shift": float(np.median(v["dthr"])),
                                              "median_fpr_iqr_over_windows": float(np.median(v["fpr_iqr"])), "n_instances": len(v["fpr"])}
                                         for c, v in per.items() if v["fpr"]}}
        log.info("F %s: %s", gname, {c: (round(100 * F[gname][c]["median_fpr"], 2), round(100 * F[gname][c]["median_recall"], 1))
                                     for c in per if c in F[gname]})
    if U.exists():
        yv = Zu["yv"].astype(bool)
        per = {str(c): {"fpr": [], "rec": []} for c in (0.0, 0.01, 0.05, 0.10)}
        for i in range(len(Zu["models"])):
            d1 = Zu[f"u_{days[0]}"][i].astype(np.float64)
            allb = np.concatenate([d1[1000:]] + [Zu[f"u_{d}"][i] for d in days[1:]]).astype(np.float64)
            att = Zu["sv"][i].astype(np.float64)[yv]
            for c in per:
                na = int(round(1000 * float(c))); fp, rc = [], []
                for _ in range(20):
                    ai = rng.choice(len(att), na, replace=False) if na else np.array([], int)
                    t = conformal(np.r_[d1[:1000 - na], att[ai]])
                    ma = np.ones(len(att), bool); ma[ai] = False
                    fp.append(float((allb > t).mean())); rc.append(float((att[ma] > t).mean()))
                per[c]["fpr"].append(np.median(fp)); per[c]["rec"].append(np.median(rc))
        F["real_home"] = {c: {"median_fpr": float(np.median(v["fpr"])), "max_fpr": float(np.max(v["fpr"])),
                              "median_lab_recall": float(np.median(v["rec"])), "min_lab_recall": float(np.min(v["rec"]))}
                          for c, v in per.items()}
        log.info("F real_home: %s", F["real_home"])
    out["F_contamination"] = F
    out["instances"] = [{k: v for k, v in r.items() if k != "i"} for r in inst]
    save_json(out, Path(str(P.v3).replace("outputs_v3", "outputs_v4")) / "regime_v4.json")


if __name__ == "__main__":
    main()
