"""v4: label-free regime signals re-scored against the v4 Case 2 definition (R*_1% < 0.5).

Signals per (setting, model) come from the v3 runs (regime_check.json, regime_loco.json: domain AUC, score KS,
conformal abstention, ATC, disagreement); R* per (setting, model) comes from outputs_v4/regime_v4.json (scripts/67).
Reported per signal: setting-level AUC (median over models) for Case 2 with bootstrap CIs over settings and over capture
clusters, and the AUC within IoT-23 only (the dataset where both cases occur). Also logs the attack shares."""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("v4_signals")
_s = importlib.util.spec_from_file_location("a67", Path(__file__).with_name("67_regime_v4_analysis.py"))
a67 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a67)


def main():
    v3 = load_json("D:/shield_run/outputs_v3/regime_check.json")["instances"] + \
        load_json("D:/shield_run/outputs_v3/regime_loco.json")["instances"]
    v4 = load_json("D:/shield_run/outputs_v4/regime_v4.json")
    rs = {(r["setting"], r["model"]): r["rstar"] for r in v4["instances"]}
    sig = ["score_ks", "atc", "disagree", "dom_auc", "conf_abst"]
    settings = sorted(v4["settings"])
    rng = np.random.default_rng(0)
    out = {}
    for p in sig:
        x = np.array([np.median([r[p] for r in v3 if r["setting"] == s]) for s in settings])
        lab = np.array([v4["settings"][s]["rstar"] < 0.5 for s in settings])
        grp = [v4["settings"][s]["cluster"] for s in settings]
        res = a67.auc_boot(lab, x, grp, rng)
        io = np.array([s.startswith("iot23") for s in settings])
        res["auc_iot23_only"] = float(roc_auc_score(lab[io], x[io])) if 0 < lab[io].sum() < io.sum() else None
        miss = [k for k in {(r["setting"], r["model"]) for r in v3} if k not in rs]
        res["unmatched_instances"] = len(miss)
        out[p] = res
        log.info("%-10s AUC %.2f naive %s cluster %s | IoT-23 only %s", p, res["auc"], np.round(res["ci_naive"], 2),
                 np.round(res["ci_cluster"], 2), res["auc_iot23_only"] and round(res["auc_iot23_only"], 2))
    # attack shares of validation and deployment data, and KS on benign-only / attack-only scores
    from scipy.stats import ks_2samp, spearmanr
    shares = {}
    for s in settings:
        Z = np.load(f"D:/shield_run/outputs_v4/regime/{s}.npz", allow_pickle=True)
        yv, yt = Z["yv"].astype(bool), Z["yt"].astype(bool)
        kb = [ks_2samp(Z["sv"][i][~yv], Z["st"][i][~yt]).statistic for i in range(len(Z["models"]))]
        ka = [ks_2samp(Z["sv"][i][yv], Z["st"][i][yt]).statistic for i in range(len(Z["models"]))]
        shares[s] = {"val_attack_share": float(yv.mean()), "dep_attack_share": float(yt.mean()),
                     "ks_benign_only": float(np.median(kb)), "ks_attack_only": float(np.median(ka)),
                     "ks_full": v4["settings"][s]["ks"]}
    d = np.array([abs(v["val_attack_share"] - v["dep_attack_share"]) for v in shares.values()])
    k = np.array([v["ks_full"] for v in shares.values()])
    out["attack_shares"] = shares
    out["spearman_ks_vs_share_gap"] = float(spearmanr(k, d).statistic)
    log.info("Spearman(KS, |attack share gap|) = %.2f", out["spearman_ks_vs_share_gap"])
    save_json(out, "D:/shield_run/outputs_v4/signals_v4.json")


if __name__ == "__main__":
    main()
