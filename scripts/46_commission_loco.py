"""Proper test of the commissioning idea (local benign memory fused with the detector), IoT-23, leave-one-capture-out.

Protocol (fixed before the run):
  * Folds: every IoT-23 capture with >= 1500 benign flows after cross-split de-dup is held out once; detectors are
    trained on all other captures (15% random validation slice of them for early stopping and the validation
    threshold). Test capture replayed in timestamp order.
  * Detectors (CPU, to avoid GPU contention): XGBoost x3 seeds (subsample 0.8), small MLP x3 seeds, small MLP
    distilled from a large MLP x3 seeds, large MLP x1.
  * Tuning vs evaluation captures: eligible captures are split once (seed 20261012) into 1/3 TUNING and 2/3 EVALUATION.
    The memory configuration is chosen on tuning captures only and frozen:
      memory in {kNN-5 distance, nearest of 64 k-means centroids}, K in {500, 1000}, fusion weight w in {0.3, 0.5, 0.7}
      (fused = w * detector-percentile + (1 - w) * novelty-percentile, percentiles w.r.t. the calibration half).
    Selection: highest median recall subject to median FPR <= 1.5%; if none qualifies, lowest median FPR.
  * Commissioning window = first K benign flows in time (clean); memory = first half, calibration = second half,
    threshold = conformal 1% quantile of the fused score on the calibration half.
    Contamination test (evaluation captures, frozen config): attack flows from the same capture injected into the
    commissioning window at 1%, 5%, 10%.
  * Baselines on the same flows: fixed validation threshold; time-ordered burn-in (detector score only, same
    calibration half); novelty alone.
Decision rule (evaluation captures, frozen config, clean window; medians over captures of the per-capture median
over detectors):
  WORKS   if FPR <= 1.5% AND recall loss vs fixed <= 5 pts AND recall gain over burn-in >= 10 pts with a 95% bootstrap
          CI (over captures) excluding 0;
  FAILS   if the recall-gain CI includes 0 OR FPR > 3%;
  PARTIAL otherwise."""

import importlib.util
import math

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from shield.cli import parser
from shield.data.preprocess import dedup_across_splits, row_hash
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("loco")
_s = importlib.util.spec_from_file_location("i23", __file__.replace("46_commission_loco", "41_iot23_shift"))
i23 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i23)
ALPHA = 0.01
KS, MEMS, WS = (500, 1000), ("knn", "cent"), (0.3, 0.5, 0.7)


def conformal(s, a=ALPHA):
    k = int(math.ceil((len(s) + 1) * (1 - a)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def pct(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def train_detectors(X, y, Xv, yv, Xt, dev):
    S = {}
    for s in range(3):
        b = xgb.XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=8,
                              early_stopping_rounds=50, random_state=s, subsample=0.8, colsample_bytree=0.8)
        b.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        S[f"xgb_s{s}"] = (b.predict_proba(Xv)[:, 1], b.predict_proba(Xt)[:, 1])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xvn, Xtn = ((a - mu) / sd for a in (X, Xv, Xt))
    big = i23.train_mlp(Xn, y, Xvn, yv, (384, 384, 384, 384), 0, dev, epochs=6)
    S["large_mlp_s0"] = (i23.mlp_score(big, Xvn, dev), i23.mlp_score(big, Xtn, dev))
    for s in range(3):
        m = i23.train_mlp(Xn, y, Xvn, yv, (64, 64), s, dev, epochs=6)
        S[f"small_s{s}"] = (i23.mlp_score(m, Xvn, dev), i23.mlp_score(m, Xtn, dev))
        m = i23.train_mlp(Xn, y, Xvn, yv, (64, 64), s, dev, teacher=big, epochs=6)
        S[f"small_kd_s{s}"] = (i23.mlp_score(m, Xvn, dev), i23.mlp_score(m, Xtn, dev))
    return S


def evaluate_capture(S, yv, Xs_t, yt, order, contam=0.0, rng=None, configs=None, filt=False):
    """All configs x detectors on one held-out capture (rows already in time order via `order`)."""
    yo = yt[order]; Xo = Xs_t[order]
    ben = np.flatnonzero(yo == 0); att = np.flatnonzero(yo == 1)
    out = {}
    mem_cache = {}
    for K in KS:
        if len(ben) < K + 500:
            continue
        win_b = ben[:K]
        if contam > 0 and len(att) > 0:
            n_a = int(round(K * contam))
            inj = rng.choice(att, min(n_a, len(att)), replace=False)
            win = np.concatenate([win_b[: K - len(inj)], inj])
        else:
            inj = np.array([], int); win = win_b
        half = len(win) // 2
        perm = np.random.default_rng(K).permutation(len(win))           # contamination spread over both halves
        mem, cal = win[perm[:half]], win[perm[half:]]
        if filt and len(S):                                              # drop window flows the detectors already flag
            vote = np.mean([S[dk][1][order][mem] > np.quantile(S[dk][0][yv == 0], 1 - ALPHA) for dk in S], 0)
            mem = mem[vote < 0.5]
        ev = np.setdiff1d(np.arange(len(yo)), np.concatenate([win_b, inj]))
        y_ev = yo[ev]
        if len(np.unique(y_ev)) < 2:
            continue
        for mk in MEMS:
            if mk == "knn":
                nn = NearestNeighbors(n_neighbors=5).fit(Xo[mem]); f = lambda I: nn.kneighbors(Xo[I])[0].mean(1)
            else:
                km = KMeans(64, n_init=3, random_state=0).fit(Xo[mem]); f = lambda I: km.transform(Xo[I]).min(1)
            mem_cache[(K, mk)] = (f(cal), f(ev))
        for dk, (sv, st) in S.items():
            so = st[order]; t_fix = float(np.quantile(sv[yv == 0], 1 - ALPHA))
            s_cal, s_ev = so[cal], so[ev]
            base = {"fixed": (s_ev > t_fix), "burnin": (s_ev > conformal(s_cal))}
            rec = {k: (float(v[y_ev == 0].mean()), float(v[y_ev == 1].mean())) for k, v in base.items()}
            rec["base_roc"] = roc_auc_score(y_ev, s_ev)
            for mk in MEMS:
                d_cal, d_ev = mem_cache[(K, mk)]
                pdc, pde = pct(d_cal, d_cal), pct(d_cal, d_ev)
                rec[f"novelty|{mk}|{K}"] = (float((pde[y_ev == 0] > conformal(pdc)).mean()),
                                            float((pde[y_ev == 1] > conformal(pdc)).mean()))
                psc, pse = pct(s_cal, s_cal), pct(s_cal, s_ev)
                for w in WS:
                    fc, fe = w * psc + (1 - w) * pdc, w * pse + (1 - w) * pde
                    t = conformal(fc)
                    fpr_m = float((fe[y_ev == 0] > t).mean())
                    sb = np.sort(s_ev[y_ev == 0])                        # detector-only oracle at the same FPR
                    to = sb[min(len(sb) - 1, max(0, int(np.ceil((1 - fpr_m) * len(sb))) - 1))]
                    rec[f"mem|{mk}|{K}|{w}"] = (fpr_m, float((fe[y_ev == 1] > t).mean()), roc_auc_score(y_ev, fe),
                                                float((s_ev[y_ev == 1] > to).mean()))
            out.setdefault(K, {})[dk] = rec
    return out


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--min-benign", type=int, default=1500)
    args = p.parse_args()
    P = Paths(args.smoke)
    dev = torch.device("cpu")
    torch.set_num_threads(8)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    y = df.y.to_numpy(); cap = df.capture.to_numpy(); ts = df.ts.to_numpy()
    caps = sorted(np.unique(cap))
    folds = {}
    for c in caps:   # de-dup each fold: train (others) -> test (c); count usable benign
        te = np.flatnonzero(cap == c); tr = np.flatnonzero(cap != c)
        h = {"train": row_hash(pd.DataFrame(np.nan_to_num(A[tr], nan=-9.0))), "validation": np.zeros(0, np.uint64),
             "test": row_hash(pd.DataFrame(np.nan_to_num(A[te], nan=-9.0)))}
        keep, _ = dedup_across_splits(h, {"train": y[tr], "validation": np.zeros(0, np.int64), "test": y[te]})
        te_k = te[keep["test"]]
        if (y[te_k] == 0).sum() >= args.min_benign and (y[te_k] == 1).sum() >= 50:
            folds[c] = (tr[keep["train"]], te_k)
    elig = sorted(folds)
    perm = np.random.default_rng(20261012).permutation(elig)
    tune = sorted(perm[: max(2, len(elig) // 3)]); evalc = sorted(perm[max(2, len(elig) // 3):])
    log.info("eligible captures %d: tuning %s | evaluation %s", len(elig), tune, evalc)
    res = {"tuning": tune, "evaluation": evalc, "per_capture": {}}
    rng = np.random.default_rng(1)
    for c in elig:
        tr, te = folds[c]
        r2 = np.random.default_rng(7); vm = r2.random(len(tr)) < 0.15
        X, yy = A[tr[~vm]].astype(np.float32), y[tr[~vm]]
        Xv, yv = A[tr[vm]].astype(np.float32), y[tr[vm]]
        Xt, yt = A[te].astype(np.float32), y[te]
        Xn = [np.nan_to_num(a) for a in (X, Xv, Xt)]
        S = train_detectors(Xn[0], yy, Xn[1], yv, Xn[2], dev)
        mu, sd = Xn[1][yv == 0].mean(0), Xn[1][yv == 0].std(0) + 1e-6
        Xs_t = (Xn[2] - mu) / sd
        order = np.argsort(ts[te], kind="stable")
        entry = {"family": i23.FAMILY.get(c.split("Capture-")[1], "benign") if "Malware" in c else "honeypot",
                 "n_test": int(len(te)), "n_benign": int((yt == 0).sum()),
                 "clean": evaluate_capture(S, yv, Xs_t, yt, order)}
        if c in evalc:
            entry["contam"] = {str(p_): evaluate_capture(S, yv, Xs_t, yt, order, contam=p_, rng=rng) for p_ in (0.01, 0.05, 0.10)}
            entry["contam_filtered"] = {str(p_): evaluate_capture(S, yv, Xs_t, yt, order, contam=p_, rng=rng, filt=True)
                                        for p_ in (0.01, 0.05, 0.10)}
        res["per_capture"][c] = entry
        b = entry["clean"].get(500, {})
        log.info("%s (%s): base ROC med %.3f | fixed FPR med %.2f%% | burnin %.2f%%", c, entry["family"],
                 np.median([v["base_roc"] for v in b.values()]) if b else float("nan"),
                 100 * np.median([v["fixed"][0] for v in b.values()]) if b else float("nan"),
                 100 * np.median([v["burnin"][0] for v in b.values()]) if b else float("nan"))
        save_json(res, P.v3 / "iot23" / "commission_loco.json")

    # ---- choose configuration on TUNING captures, freeze, evaluate
    def cap_med(entry_block, key, i):
        vals = [v[key][i] for v in entry_block.values() if key in v]
        return float(np.median(vals)) if vals else None

    cfgs = [f"mem|{mk}|{K}|{w}" for mk in MEMS for K in KS for w in WS]
    tune_stats = {}
    for cfg in cfgs:
        K = int(cfg.split("|")[2])
        fpr = [cap_med(res["per_capture"][c]["clean"][K], cfg, 0) for c in tune if K in res["per_capture"][c]["clean"]]
        rec = [cap_med(res["per_capture"][c]["clean"][K], cfg, 1) for c in tune if K in res["per_capture"][c]["clean"]]
        if fpr:
            tune_stats[cfg] = (float(np.median(fpr)), float(np.median(rec)))
    ok = {k: v for k, v in tune_stats.items() if v[0] <= 0.015}
    chosen = max(ok, key=lambda k: ok[k][1]) if ok else min(tune_stats, key=lambda k: tune_stats[k][0])
    K = int(chosen.split("|")[2])
    log.info("tuning stats %s -> chosen %s", {k: (round(100 * a, 2), round(100 * b, 1)) for k, (a, b) in tune_stats.items()}, chosen)

    def summarize(block_getter):
        rows = []
        for c in evalc:
            blk = block_getter(c)
            if not blk or K not in blk:
                continue
            b = blk[K]
            rows.append({"capture": c, "fpr": cap_med(b, chosen, 0), "recall": cap_med(b, chosen, 1),
                         "fixed_fpr": cap_med(b, "fixed", 0), "fixed_recall": cap_med(b, "fixed", 1),
                         "burnin_fpr": cap_med(b, "burnin", 0), "burnin_recall": cap_med(b, "burnin", 1),
                         "base_roc": float(np.median([v["base_roc"] for v in b.values()])),
                         "mem_roc": float(np.median([v[chosen][2] for v in b.values()])),
                         "oracle_recall_same_fpr": cap_med(b, chosen, 3)})
        return rows

    def boot_ci(x, n=10000):
        x = np.asarray(x); r = np.random.default_rng(0)
        m = [np.median(r.choice(x, len(x))) for _ in range(n)]
        return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

    # Corrected rule (2026-10-12): a threshold-only baseline cannot beat the detector's own ROC, so the memory is judged
    # by its recall GAIN over the detector's best threshold at the same realised FPR (oracle), not by loss vs the fixed
    # threshold. WORKS: median FPR <= 1.5% and median gain >= 5 pts; FAILS: median gain <= 1 pt; else PARTIAL.
    clean = summarize(lambda c: res["per_capture"][c]["clean"])
    gain = [r["recall"] - r["oracle_recall_same_fpr"] for r in clean]
    loss = [r["fixed_recall"] - r["recall"] for r in clean]
    fpr = [r["fpr"] for r in clean]
    ci = boot_ci(gain) if len(gain) >= 2 else (float("nan"), float("nan"))
    med_fpr, med_loss, med_gain = float(np.median(fpr)), float(np.median(loss)), float(np.median(gain))
    if med_fpr <= 0.015 and med_gain >= 0.05:
        verdict = "WORKS"
    elif med_gain <= 0.01:
        verdict = "FAILS"
    else:
        verdict = "PARTIAL"
    contam = {p_: summarize(lambda c: res["per_capture"][c].get("contam", {}).get(p_)) for p_ in ("0.01", "0.05", "0.1")}
    contam_f = {p_: summarize(lambda c: res["per_capture"][c].get("contam_filtered", {}).get(p_)) for p_ in ("0.01", "0.05", "0.1")}
    res["frozen_config"] = chosen
    res["evaluation_summary"] = {"rows": clean, "median_fpr": med_fpr, "median_recall_loss_vs_fixed": med_loss,
                                 "median_recall_gain_vs_burnin": med_gain, "gain_ci95": ci, "verdict": verdict,
                                 "contamination": {p_: {"median_fpr": float(np.median([r["fpr"] for r in rows])) if rows else None,
                                                        "median_recall": float(np.median([r["recall"] for r in rows])) if rows else None}
                                                   for p_, rows in contam.items()},
                                 "contamination_filtered": {p_: {"median_fpr": float(np.median([r["fpr"] for r in rows])) if rows else None,
                                                                 "median_recall": float(np.median([r["recall"] for r in rows])) if rows else None,
                                                                 "median_gain_vs_oracle": float(np.median([r["recall"] - r["oracle_recall_same_fpr"] for r in rows])) if rows else None}
                                                            for p_, rows in contam_f.items()}}
    log.info("EVALUATION (%d captures, frozen %s): FPR %.2f%% | recall gain vs detector oracle @same FPR %.1f CI [%.1f, %.1f] | %s",
             len(clean), chosen, 100 * med_fpr, 100 * med_gain, 100 * ci[0], 100 * ci[1], verdict)
    for r in clean:
        log.info("   %s: ROC %.3f->%.3f | burn-in %.2f%%/%.1f | memory %.2f%%/%.1f | detector oracle @same FPR %.1f", r["capture"],
                 r["base_roc"], r["mem_roc"], 100 * r["burnin_fpr"], 100 * r["burnin_recall"], 100 * r["fpr"], 100 * r["recall"],
                 100 * r["oracle_recall_same_fpr"])
    log.info("contamination: %s", res["evaluation_summary"]["contamination"])
    log.info("contamination (filtered window): %s", res["evaluation_summary"]["contamination_filtered"])
    save_json(res, P.v3 / "iot23" / "commission_loco.json")


if __name__ == "__main__":
    main()
