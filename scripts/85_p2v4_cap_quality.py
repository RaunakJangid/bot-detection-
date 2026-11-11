"""P2 v4, experiment X11: detection quality of the cascade with the online escalation cap, all shift scenarios, and the
escalation scenarios used by the robust placement (scripts/84). CPU only, no training.

Cascades and budget as scripts/54 / 76 (students shield_s0-4 / small_mlp_kd_s0-4, experts teacher_s0 / ensemble_s0, 10%
validation budget). Per setting and seed: escalated share and macro-F1 for (a) the validation threshold (no cap), (b) the
offline cap (test-set quantile), (c) the online cap: running 90th percentile of the entropies of the previous 300 flows,
refreshed every 50 flows (time order for IoT-23, a fixed random order for CICIoMT2024, which has no flow times), and the
escalated share in every block of 1,000 flows under (a) and (c).
Summary per setting: median over seeds, range, and a 95% bootstrap interval of the median macro-F1 difference (c) - (a)
over flows (1,000 resamples, seed of the median model).
Output: outputs_v5/p2/cap_quality_v4.json (summary) and escalation_scenarios.json (per seed: e without cap, and block
shares under the online cap)."""

import json
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

B, W = 0.10, 300
OUT = Path("D:/shield_run/outputs_v5/p2")
rng = np.random.default_rng(85)
SETTINGS_IOT = ["iot23_capture_shift_CTU-IoT-Malware-Capture-1-1", "iot23_same_family_CTU-IoT-Malware-Capture-49-1",
                "iot23_same_family_CTU-IoT-Malware-Capture-7-1", "iot23_capture_shift_CTU-IoT-Malware-Capture-35-1"]


def ent(P):
    P = np.clip(P, 1e-12, 1); return -(P * np.log(P)).sum(1)


def mf1(y, p):
    return float(100 * f1_score(y, p, average="macro"))


def online(H, t_val):
    esc = np.zeros(len(H), bool); win = deque(maxlen=W); thr = t_val; cnt = 0
    for i, h in enumerate(H):
        if len(win) == W and cnt == 0:
            thr = np.quantile(np.fromiter(win, float), 1 - B); cnt = 50
        if len(win) == W:
            cnt -= 1
        esc[i] = h > (thr if len(win) == W else t_val); win.append(h)
    return esc


def blocks(esc, n=1000):
    return [float(esc[i:i + n].mean()) for i in range(0, len(esc) - n + 1, n)]


def evaluate(name, y, pe, ps, Hv, Ht, order):
    tv = np.quantile(Hv, 1 - B)
    y, pe, ps, Ht = y[order], pe[order], ps[order], Ht[order]
    e_nc = Ht > tv; e_off = Ht > np.quantile(Ht, 1 - B); e_on = online(Ht, tv)
    pred = {k: np.where(e, pe, ps) for k, e in (("nocap", e_nc), ("offline", e_off), ("online", e_on))}
    return {"e_nocap": float(e_nc.mean()), "e_online": float(e_on.mean()),
            "f1": {k: mf1(y, v) for k, v in pred.items()}, "f1_student": mf1(y, ps), "f1_expert": mf1(y, pe),
            "blocks_nocap": blocks(e_nc), "blocks_online": blocks(e_on), "_pred": pred, "_y": y}


def summarise(rows):
    f = lambda k: [r["f1"][k] for r in rows]
    med = int(np.argsort(f("online"))[len(rows) // 2])
    r = rows[med]; y = r["_y"]; d = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        d.append(mf1(y[i], r["_pred"]["online"][i]) - mf1(y[i], r["_pred"]["nocap"][i]))
    on_b = [b for x in rows for b in x["blocks_online"]]
    return {"e_nocap_median": float(np.median([x["e_nocap"] for x in rows])),
            "e_online_median": float(np.median([x["e_online"] for x in rows])),
            "online_block_p95": float(np.quantile(on_b, .95)) if on_b else None,
            "online_block_max": float(max(on_b)) if on_b else None,
            **{f"f1_{k}_median": float(np.median(f(k))) for k in ("nocap", "offline", "online")},
            "f1_online_minus_nocap_range": [float(min(np.subtract(f("online"), f("nocap")))), float(max(np.subtract(f("online"), f("nocap"))))],
            "f1_online_minus_nocap_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "f1_student_median": float(np.median([x["f1_student"] for x in rows])),
            "f1_expert_median": float(np.median([x["f1_expert"] for x in rows]))}


def main():
    P = Paths(False); out, scen = {}, {}
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test")); yt = d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"
        pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)).argmax(1)
        order = np.random.default_rng(0).permutation(len(yt))           # no flow times: one fixed random order
        rows = []
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))
            rows.append(evaluate(f"{ds}_{task}", yt, pe, Pt.argmax(1), ent(Pv), ent(Pt), order))
        key = f"{ds}_{task}"; out[key] = summarise(rows)
        scen[key] = {"e_nocap": [r["e_nocap"] for r in rows], "blocks_online": [r["blocks_online"] for r in rows]}
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()}, flush=True)
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True)
        models = list(Z["models"]); pe_all = (Z["st"][models.index("ensemble_s0")] > 0.5).astype(int)
        for c in np.unique(Z["capture"]):
            key = f"iot23_{design}_{c}"
            if key not in SETTINGS_IOT:
                continue
            m = Z["capture"] == c; yt = Z["yt"][m]; order = np.argsort(Z["ts"][m], kind="stable"); rows = []
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}"); sv, st = Z["sv"][i], Z["st"][i][m]
                rows.append(evaluate(key, yt, pe_all[m], (st > 0.5).astype(int), ent(np.c_[1 - sv, sv]), ent(np.c_[1 - st, st]), order))
            out[key] = summarise(rows)
            scen[key] = {"e_nocap": [r["e_nocap"] for r in rows], "blocks_online": [r["blocks_online"] for r in rows]}
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()}, flush=True)
    json.dump(out, open(OUT / "cap_quality_v4.json", "w"), indent=1)
    json.dump(scen, open(OUT / "escalation_scenarios.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
