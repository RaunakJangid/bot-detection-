"""P2 v4, go/no-go test of "expert-budget competition" (external review): does the quota reject flows the teacher would
have corrected, alone and after an emulated controller failover? CPU only, existing predictions.

Per flow: correctable G = student wrong and teacher right; harmful B = student right and teacher wrong.
Candidates = entropy above the validation 90th percentile (what a fixed threshold would escalate); admitted = the deployed
rule (online quantile + quota, scripts/88 quota()). Rejected = candidates not admitted.
Per setting and seed (5 seeds): shares of G and B among candidates and among rejected flows; net accuracy effect of the
rejections on all flows = (|B and rejected| - |G and rejected|) / n (positive = rejecting helped).
Failover emulation (IoT-23, time-ordered captures as controller streams): for each ordered pair (surviving controller S,
failed controller F), the two streams are merged by time since capture start, so after the failure S's quota serves both;
for F's flows: lost corrections and net effect versus F served alone with its own quota.
Settings: CICIoT2023 34-class (control), CICIoMT2024 6-class and 19-class (fixed order), IoT-23 1-1, 35-1, 49-1, 7-1.
Output: outputs_v5/p2/quota_competition.json."""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m85, m88 = _l("85_p2v4_cap_quality.py"), _l("88_p2v4_escalation_attack.py")


def counts(y, ps, pe, H, tv):
    cand = H > tv; adm = m88.quota(H, tv); rej = cand & ~adm
    G = (ps != y) & (pe == y); B = (ps == y) & (pe != y); n = len(y)
    return {"n": n, "cand": int(cand.sum()), "adm": int(adm.sum()), "rej": int(rej.sum()),
            "G_cand": int((G & cand).sum()), "B_cand": int((B & cand).sum()), "G_rej": int((G & rej).sum()),
            "B_rej": int((B & rej).sum()), "G_all": int(G.sum()), "B_all": int(B.sum()),
            "net_effect_pts": 100 * ((B & rej).sum() - (G & rej).sum()) / n}, adm


def main():
    from shield.data.datasets import load_processed
    from shield.eval.metrics import softmax
    from shield.utils.config import Paths
    P = Paths(False); res = {"settings": {}, "failover": []}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test")); yt = d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"; pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)).argmax(1)
        o = np.random.default_rng(0).permutation(len(yt)); rows = []
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))
            c, _ = counts(yt[o], Pt.argmax(1)[o], pe[o], m85.ent(Pt)[o], np.quantile(m85.ent(Pv), 0.9)); rows.append(c)
        res["settings"][f"{ds}_{task}"] = rows
        print(ds, task, {k: int(np.median([r[k] for r in rows])) for k in ("n", "cand", "rej", "G_cand", "B_cand", "G_rej", "B_rej")},
              "net", round(float(np.median([r["net_effect_pts"] for r in rows])), 3), flush=True)
    streams = {}
    for design in ("capture_shift", "same_family"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        for cap in ("CTU-IoT-Malware-Capture-1-1", "CTU-IoT-Malware-Capture-35-1", "CTU-IoT-Malware-Capture-49-1",
                    "CTU-IoT-Malware-Capture-7-1"):
            m = Z["capture"] == cap
            if not m.any():
                continue
            ordr = np.argsort(Z["ts"][m], kind="stable"); ts = Z["ts"][m][ordr]; ts = ts - ts.min(); yt = Z["yt"][m][ordr].astype(int)
            pe = (Z["st"][models.index("ensemble_s0")][m][ordr] > 0.5).astype(int); rows = []; per_seed = []
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                st = Z["st"][i][m][ordr]; H = m85.ent(np.c_[1 - st, st]); sv = Z["sv"][i]; tv = np.quantile(m85.ent(np.c_[1 - sv, sv]), 0.9)
                ps = (st > 0.5).astype(int); c, _ = counts(yt, ps, pe, H, tv); rows.append(c); per_seed.append((ps, H, tv))
            res["settings"][cap] = rows; streams[cap] = (ts, yt, pe, per_seed)
            print(cap, {k: int(np.median([r[k] for r in rows])) for k in ("n", "cand", "rej", "G_cand", "B_cand", "G_rej", "B_rej")},
                  "net", round(float(np.median([r["net_effect_pts"] for r in rows])), 3), flush=True)
    for S, F in itertools.permutations(streams, 2):
        tsS, yS, peS, segS = streams[S]; tsF, yF, peF, segF = streams[F]
        for s in range(5):
            psS, HS, tvS = segS[s]; psF, HF, tvF = segF[s]
            alone, admF = counts(yF, psF, peF, HF, tvF)
            t = np.concatenate([tsS, tsF]); order = np.argsort(t, kind="stable"); isF = np.r_[np.zeros(len(tsS), bool), np.ones(len(tsF), bool)][order]
            H = np.r_[HS, HF][order]; tv = tvS                  # surviving controller's own threshold and quota
            adm = m88.quota(H, tv)
            admF_merged = adm[isF]; Hf = np.r_[HS, HF][order][isF]
            yff, psf, pef = np.r_[yS, yF][order][isF], np.r_[psS, psF][order][isF], np.r_[peS, peF][order][isF]
            G = (psf != yff) & (pef == yff); B = (psf == yff) & (pef != yff)
            res["failover"].append({"surviving": S, "failed": F, "seed": s, "n_failed_flows": int(len(yF)),
                                    "adm_failed_alone": int(admF.sum()), "adm_failed_merged": int(admF_merged.sum()),
                                    "corrections_alone": int((G & admF).sum()), "corrections_merged": int((G & admF_merged).sum()),
                                    "harms_alone": int((B & admF).sum()), "harms_merged": int((B & admF_merged).sum()),
                                    "net_change_pts_failed": 100 * (((G & admF_merged).sum() - (B & admF_merged).sum()) -
                                                                    ((G & admF).sum() - (B & admF).sum())) / len(yF)})
    fo = res["failover"]
    print("failover pairs", len(fo) // 5, "| median corrections lost per pair",
          float(np.median([r["corrections_alone"] - r["corrections_merged"] for r in fo])),
          "| median net accuracy change for failed controller's flows (pts)", round(float(np.median([r["net_change_pts_failed"] for r in fo])), 3),
          "| range", round(min(r["net_change_pts_failed"] for r in fo), 3), round(max(r["net_change_pts_failed"] for r in fo), 3))
    json.dump(res, open(OUT / "quota_competition.json", "w"), indent=1, default=float)
    print("saved")


if __name__ == "__main__":
    main()
