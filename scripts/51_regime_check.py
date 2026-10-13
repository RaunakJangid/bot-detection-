"""Step 2: can a LABEL-FREE check tell, before re-tuning, whether ranking survived on new traffic (Case 1: threshold
shift, fixable by recalibration) or broke (Case 2)?

Instances: (shift setting, model). Settings: CICIoMT2024 official test captures (binary/category/attack tasks; binary
benign-vs-attack score), each IoT-23 test capture with >= 300 benign and >= 300 attack flows (same-family and random
capture hold-out designs; within-capture), CICIoT2023 random split (no-shift control).
Ground truth (needs labels, used only to evaluate the predictors): target ROC-AUC benign vs attack; Case 2 := AUC < 0.9.
Label-free predictors (use labelled VALIDATION data + UNLABELLED target flows only):
  dom_auc    domain classifier AUC, validation flows vs target flows (features; 2-fold LightGBM; 20k each)
  score_ks   Kolmogorov-Smirnov distance between validation and target attack-score distributions
  conf_abst  conformal abstention: share of target flows whose 95% split-conformal prediction set (non-conformity
             1 - p(true class), calibrated on validation) is not a single class (Ahmed et al. 2026: the reliable signal)
  atc        Average Thresholded Confidence (Garg et al. 2022) predicted accuracy; we report 1 - ATC so that higher
             means "worse" for every predictor
  disagree   share of target flows on which this model and a reference model of another family disagree (at 0.5)
Evaluation: per predictor, AUC for separating Case 2 from Case 1 over instances, and Spearman correlation with target
ROC-AUC; also at setting level (median over models) because instances within a setting are correlated."""

import glob
import os

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("regime")


def dom_auc(Xa, Xb, rng):
    import lightgbm as lgb
    k = min(len(Xa), len(Xb), 20_000)
    M = np.vstack([Xa[rng.choice(len(Xa), k, replace=False)], Xb[rng.choice(len(Xb), k, replace=False)]])
    y = np.r_[np.zeros(k), np.ones(k)]; f = rng.integers(0, 2, 2 * k); s = np.zeros(2 * k)
    for i in (0, 1):
        s[f == i] = lgb.LGBMClassifier(n_estimators=150, verbose=-1).fit(M[f != i], y[f != i]).predict_proba(M[f == i])[:, 1]
    return float(roc_auc_score(y, s))


def predictors(sv, yv, st, sref_t, dom):
    pv = np.c_[1 - sv, sv]; pt = np.c_[1 - st, st]
    nc = 1 - pv[np.arange(len(yv)), yv]                                    # split-conformal on validation
    q = np.quantile(nc, 0.95)
    set_size = ((1 - pt) <= q).sum(1)
    acc_v = float(((sv > 0.5).astype(int) == yv).mean())
    cv, ct = pv.max(1), pt.max(1)
    t = np.quantile(cv, 1 - acc_v)                                          # ATC threshold: share above = val acc
    return {"dom_auc": dom, "score_ks": float(ks_2samp(sv, st).statistic), "conf_abst": float((set_size != 1).mean()),
            "atc": float(1 - (ct > t).mean()), "disagree": float(((st > 0.5) != (sref_t > 0.5)).mean())}


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    rng = np.random.default_rng(0)
    inst = []
    # --- CICIoMT (shift) and CICIoT (control): cached logits, binary attack score = 1 - p(benign)
    for ds, task, setting in (("ciciomt", "binary", "ciciomt_binary"), ("ciciomt", "category", "ciciomt_category"),
                              ("ciciomt", "attack", "ciciomt_attack"), ("ciciot", "class34", "ciciot_control")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        ben = next(i for i, c in enumerate(d.classes) if c in ("Benign", "BenignTraffic"))
        yv = (d.splits["validation"].y != ben).astype(int); yt = (d.splits["test"].y != ben).astype(int)
        iv = rng.choice(len(yv), min(len(yv), 200_000), replace=False); it = rng.choice(len(yt), min(len(yt), 200_000), replace=False)
        Xv = np.asarray(d.splits["validation"].X)[iv]; Xt = np.asarray(d.splits["test"].X)[it]
        dom = dom_auc(Xv, Xt, rng)
        lc = P.v3 / "logits" / f"{ds}_{task}"
        models = sorted(os.path.basename(f)[:-len("_validation.npy")] for f in glob.glob(str(lc / "*_validation.npy"))
                        if os.path.exists(str(f).replace("_validation", "_test")))
        S = {m: (1 - softmax(np.load(lc / f"{m}_validation.npy").astype(np.float64)[iv])[:, ben],
                 1 - softmax(np.load(lc / f"{m}_test.npy").astype(np.float64)[it])[:, ben]) for m in models}
        ref = "teacher_s0" if "teacher_s0" in S else models[0]
        for m, (sv, st) in S.items():
            r = predictors(sv, yv[iv], st, S["xgboost_s0" if m != "xgboost_s0" and "xgboost_s0" in S else ref][1], dom)
            r.update({"setting": setting, "model": m, "target_auc": float(roc_auc_score(yt[it], st))})
            inst.append(r)
        log.info("%s: %d models, dom_auc %.3f", setting, len(S), dom)
    # --- cross-dataset (binary, shared features/classes): train on one CIC dataset, apply to the other (both directions)
    import xgboost as xgb
    from shield.data.datasets import stratified_sample
    xd = {k: load_processed(P.processed(k), "binary") for k in ("xciciot", "xciciomt")}
    for src, dst in (("xciciot", "xciciomt"), ("xciciomt", "xciciot")):
        a, b = xd[src], xd[dst]
        tr = a.splits["train"]; it = stratified_sample(tr.y, 1_000_000, rng, 1000)
        X, y = np.asarray(tr.X[it], np.float32), tr.y[it]
        Xv = np.asarray(a.splits["validation"].X, np.float32); yv_ = a.splits["validation"].y
        Xt = np.asarray(b.splits["test"].X, np.float32); yt_ = b.splits["test"].y
        ben_a = a.classes.index("Benign"); ben_b = b.classes.index("Benign")
        yvb, ytb = (yv_ != ben_a).astype(int), (yt_ != ben_b).astype(int)
        iv = rng.choice(len(yvb), min(len(yvb), 200_000), replace=False); itt = rng.choice(len(ytb), min(len(ytb), 200_000), replace=False)
        dom = dom_auc(Xv[iv], Xt[itt], rng)
        S = {}
        for s in range(3):
            c = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=8,
                                  random_state=s, subsample=0.8, colsample_bytree=0.8).fit(X, (y != ben_a).astype(int))
            S[f"xgb_s{s}"] = (c.predict_proba(Xv[iv])[:, 1], c.predict_proba(Xt[itt])[:, 1])
        from sklearn.neural_network import MLPClassifier
        for s in range(2):
            m = MLPClassifier((64, 64), max_iter=30, random_state=s).fit(X[:300_000], (y[:300_000] != ben_a).astype(int))
            S[f"mlp_s{s}"] = (m.predict_proba(Xv[iv])[:, 1], m.predict_proba(Xt[itt])[:, 1])
        for mk, (sv, st) in S.items():
            ref = S["mlp_s0"][1] if mk.startswith("xgb") else S["xgb_s0"][1]
            r = predictors(sv, yvb[iv], st, ref, dom)
            r.update({"setting": f"cross_{src}_to_{dst}", "model": mk, "target_auc": float(roc_auc_score(ytb[itt], st))})
            inst.append(r)
        log.info("cross %s -> %s: dom_auc %.3f", src, dst, dom)
    # --- IoT-23 per capture (within-capture), both hold-out designs, plus the random-split control
    for design in ("same_family", "capture_shift", "random_control"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz")
        models, sv_all, st_all, yv, yt, cap, Xv, Xt = (list(Z["models"]), Z["sv"], Z["st"], Z["yv"], Z["yt"], Z["capture"],
                                                       Z["Xv"], Z["Xt"])
        xi = models.index("xgboost_s0")
        groups = np.unique(cap) if design != "random_control" else ["all"]
        for c in groups:
            m_ = cap == c if design != "random_control" else np.ones(len(yt), bool)
            if (yt[m_] == 0).sum() < 100 or (yt[m_] == 1).sum() < 100:
                continue
            dom = dom_auc(Xv, Xt[m_], rng)
            for mi, m in enumerate(models):
                ref = st_all[xi][m_] if mi != xi else st_all[models.index("large_mlp_s0")][m_]
                r = predictors(sv_all[mi], yv, st_all[mi][m_], ref, dom)
                r.update({"setting": f"iot23_{design}_{c}", "model": m, "target_auc": float(roc_auc_score(yt[m_], st_all[mi][m_]))})
                inst.append(r)
            log.info("iot23 %s %s: dom_auc %.3f", design, c, dom)
    # --- evaluation
    preds = ["dom_auc", "score_ks", "conf_abst", "atc", "disagree"]
    case2 = np.array([r["target_auc"] < 0.9 for r in inst]); tauc = np.array([r["target_auc"] for r in inst])
    out = {"instances": inst, "n_instances": len(inst), "n_case2": int(case2.sum()), "eval": {}}
    settings = sorted({r["setting"] for r in inst})
    for p in preds:
        v = np.array([r[p] for r in inst])
        sv_ = np.array([np.median([r[p] for r in inst if r["setting"] == s]) for s in settings])
        st_ = np.array([np.median([r["target_auc"] for r in inst if r["setting"] == s]) for s in settings])
        out["eval"][p] = {
            "auc_case2_instances": float(roc_auc_score(case2, v)) if 0 < case2.sum() < len(case2) else None,
            "spearman_vs_target_auc_instances": float(spearmanr(v, tauc).statistic),
            "auc_case2_settings": float(roc_auc_score(st_ < 0.9, sv_)) if 0 < (st_ < 0.9).sum() < len(st_) else None,
            "spearman_settings": float(spearmanr(sv_, st_).statistic)}
        log.info("%-10s case-2 AUC (instances) %s | Spearman vs target AUC %.2f | settings: AUC %s, Spearman %.2f", p,
                 out["eval"][p]["auc_case2_instances"] and round(out["eval"][p]["auc_case2_instances"], 3),
                 out["eval"][p]["spearman_vs_target_auc_instances"],
                 out["eval"][p]["auc_case2_settings"] and round(out["eval"][p]["auc_case2_settings"], 3), out["eval"][p]["spearman_settings"])
    out["settings"] = {s: {"median_target_auc": float(np.median([r["target_auc"] for r in inst if r["setting"] == s])),
                           **{p: float(np.median([r[p] for r in inst if r["setting"] == s])) for p in preds}} for s in settings}
    log.info("n instances %d (case 2: %d), settings %d", len(inst), int(case2.sum()), len(settings))
    save_json(out, P.v3 / "regime_check.json")


if __name__ == "__main__":
    main()
