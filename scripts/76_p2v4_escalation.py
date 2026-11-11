"""P2 v4, experiments X2 and X3 from saved scores (no training). Output: outputs_v5/p2/escalation_v4.json.

X2  Same cascades and budgets as scripts/54 Part A (students shield_s0-4 / small_mlp_kd_s0-4; experts teacher_s0 /
    ensemble_s0; entropy threshold for a 10% validation budget), but
      macro-F1 for every setting (IoT-23 too: both classes; v3 reported the attack-class F1 there), and
      IoT-23 escalation re-weighted to the true class mix of the capture: the sampler of scripts/41 kept benign and
      malicious flows at different rates (w = seen / kept per capture and label, outputs_v5/iot23_sampling_v5.json), so
      e_true = (e_b n_b w_b + e_m n_m w_m) / (n_b w_b + n_m w_m), with e_b, e_m the escalated shares among the test
      flows of each label.
X3  Online cap (the deployable form of Eq. (11)): flows of each IoT-23 test capture in time order; flow t is escalated
    if its entropy exceeds the (1 - b) quantile of the entropies of the previous W flows (W = 300, 1,000, 5,000; the
    validation threshold until W flows have been seen). Reported: overall escalated share, share per block of 1,000
    flows (median, 95th percentile, max), macro-F1, against the offline test-set quantile of v3."""

import json
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

B = 0.10
OUT = Path("D:/shield_run/outputs_v5/p2"); OUT.mkdir(parents=True, exist_ok=True)
SAMP = json.load(open("D:/shield_run/outputs_v5/iot23_sampling_v5.json"))


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def mf1(y, p):
    return float(100 * f1_score(y, p, average="macro"))


def online_escalate(H_time, t_val, W, b=B):
    esc = np.zeros(len(H_time), bool); win = deque(maxlen=W); buf = None; recompute = 0
    for i, h in enumerate(H_time):
        if len(win) < W:
            t = t_val
        else:
            if recompute == 0:                      # refresh the window quantile every 50 flows (as a gateway would)
                buf = np.quantile(np.fromiter(win, float), 1 - b)
                recompute = 50
            recompute -= 1; t = buf
        esc[i] = h > t; win.append(h)
    return esc


def blocks(esc, n=1000):
    s = [esc[i:i + n].mean() for i in range(0, len(esc) - n + 1, n)]
    return {"median": float(np.median(s)), "p95": float(np.quantile(s, .95)), "max": float(np.max(s)), "n_blocks": len(s)} if s else None


def main():
    P = Paths(False)
    out = {"x2": {}, "x3": {}}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test")); yt = d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"
        pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)).argmax(1)
        rows = []
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64)); ps = Pt.argmax(1)
            Hv, Ht = entropy(Pv), entropy(Pt); esc = Ht > np.quantile(Hv, 1 - B); cap = Ht > np.quantile(Ht, 1 - B)
            rows.append({"e": float(esc.mean()), "f1_planned": mf1(yt, np.where(esc, pe, ps)), "f1_capped": mf1(yt, np.where(cap, pe, ps)),
                         "f1_student": mf1(yt, ps), "f1_expert": mf1(yt, pe)})
        out["x2"][f"{ds}_{task}"] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        print(ds, task, out["x2"][f"{ds}_{task}"], flush=True)
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True)
        models = list(Z["models"]); yt_all, cap_all, ts_all = Z["yt"], Z["capture"], Z["ts"]
        pe_all = (Z["st"][models.index("ensemble_s0")] > 0.5).astype(int)
        for c in np.unique(cap_all):
            m = cap_all == c
            if (yt_all[m] == 0).sum() < 100 or (yt_all[m] == 1).sum() < 100:
                continue
            yt = yt_all[m]; pe = pe_all[m]; o = np.argsort(ts_all[m], kind="stable")
            wb = SAMP[c]["seen_benign"] / max(SAMP[c]["kept_benign"], 1); wm = SAMP[c]["seen_malicious"] / max(SAMP[c]["kept_malicious"], 1)
            rows, x3 = [], {str(W): [] for W in (300, 1000, 5000)}
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                sv, st = Z["sv"][i], Z["st"][i][m]
                Hv = entropy(np.c_[1 - sv, sv]); Ht = entropy(np.c_[1 - st, st]); ps = (st > 0.5).astype(int)
                tv = np.quantile(Hv, 1 - B); esc = Ht > tv; capm = Ht > np.quantile(Ht, 1 - B)
                eb, em = esc[yt == 0].mean(), esc[yt == 1].mean(); nb, nm = (yt == 0).sum(), (yt == 1).sum()
                rows.append({"e": float(esc.mean()), "e_true_mix": float((eb * nb * wb + em * nm * wm) / (nb * wb + nm * wm)),
                             "f1_planned": mf1(yt, np.where(esc, pe, ps)), "f1_capped": mf1(yt, np.where(capm, pe, ps)),
                             "f1_student": mf1(yt, ps), "f1_expert": mf1(yt, pe),
                             "f1_planned_attack": float(100 * f1_score(yt, np.where(esc, pe, ps)))})
                for W in (300, 1000, 5000):
                    eo = online_escalate(Ht[o], tv, W); po = np.where(eo, pe[o], ps[o])
                    x3[str(W)].append({"e": float(eo.mean()), "blocks": blocks(eo), "f1": mf1(yt[o], po)})
            key = f"iot23_{design}_{c}"
            out["x2"][key] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
            out["x2"][key].update({"n_test": int(m.sum()), "sample_attack_share": float(yt.mean())})
            out["x3"][key] = {W: {"e_median": float(np.median([r["e"] for r in v])),
                                  "block_p95_median": float(np.median([r["blocks"]["p95"] for r in v if r["blocks"]])) if any(r["blocks"] for r in v) else None,
                                  "block_max": float(max(r["blocks"]["max"] for r in v if r["blocks"])) if any(r["blocks"] for r in v) else None,
                                  "f1_median": float(np.median([r["f1"] for r in v]))} for W, v in x3.items()}
            print(key, out["x2"][key], out["x3"][key], flush=True)
    json.dump(out, open(OUT / "escalation_v4.json", "w"), indent=1)
    print("saved", OUT / "escalation_v4.json")


if __name__ == "__main__":
    main()
