"""P2 v4, experiment X12: escalation flooding (the threat of Al-Omar et al., Future Internet 18:585, 2026) against the
cascade with and without the online cap. CPU only, no training.

Attack model: the attacker injects flows that the student finds maximally uncertain. Proxy (mimicry): copies of real
test flows of the same setting drawn from the top 10% of student entropy, with their own student and expert
predictions and labels. Injection ratio rho = injected flows per genuine flow, rho in {0.1, 0.25, 0.5, 1, 2, 3.2}
(3.2 drives the uncapped escalated share to about 79%, the strongest level reported by Al-Omar et al.). Two patterns:
'spread' (injected flows at uniformly random positions) and 'burst' (all injected flows in one contiguous window that
starts at 30% of the stream).
Settings and cascades as scripts/85: CICIoMT2024 6-class and 19-class (teacher expert), IoT-23 1-1 (ensemble expert),
5 student seeds, 10% validation budget, online cap = running 90th percentile of the previous 300 entropies.
Third rule 'quota': the online cap plus an admission quota of at most 30 escalations among the previous 300 flows
(capacity-aware admission). Per setting, rho, pattern and seed: escalated share over all flows (nocap, online cap), worst 1,000-flow block share,
capacity ratio mu_c(e) / mu_c(0.10) at batch 256 for the realised share (worst block), macro-F1 on the genuine flows
for nocap, online cap, student only (floor) and no attack (reference).
Output: outputs_v5/p2/escalation_attack.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2")
RHOS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.2)
s85 = importlib.util.spec_from_file_location("m85", Path(__file__).with_name("85_p2v4_cap_quality.py"))
m85 = importlib.util.module_from_spec(s85); s85.loader.exec_module(m85)


def streams(P):
    """yield setting, seed, (y, pe, ps, Hv, Ht) in stream order."""
    for task in ("category", "attack"):
        d = load_processed(P.processed("ciciomt"), task, splits=("validation", "test")); yt = d.splits["test"].y
        lc = P.v3 / "logits" / f"ciciomt_{task}"
        pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)).argmax(1)
        o = np.random.default_rng(0).permutation(len(yt))
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))
            yield f"ciciomt_{task}", s, (yt[o], pe[o], Pt.argmax(1)[o], m85.ent(Pv), m85.ent(Pt)[o])
    Z = np.load(P.v3 / "iot23" / "scores_capture_shift.npz", allow_pickle=True)
    models = list(Z["models"]); m = Z["capture"] == "CTU-IoT-Malware-Capture-1-1"
    o = np.argsort(Z["ts"][m], kind="stable"); yt = Z["yt"][m][o]
    pe = (Z["st"][models.index("ensemble_s0")][m] > 0.5).astype(int)[o]
    for s in range(5):
        i = models.index(f"small_mlp_kd_s{s}"); sv, st = Z["sv"][i], Z["st"][i][m][o]
        yield "iot23_1-1", s, (yt, pe, (st > 0.5).astype(int), m85.ent(np.c_[1 - sv, sv]), m85.ent(np.c_[1 - st, st]))


def quota(H, tv, q=30, w=300):
    """online cap plus admission quota: at most q escalations among the previous w flows."""
    base = m85.online(H, tv); esc = np.zeros(len(H), bool); hist = np.zeros(len(H) + 1, int)
    for i in range(len(H)):
        if base[i] and hist[i] - hist[max(0, i - w)] < q:
            esc[i] = True
        hist[i + 1] = hist[i] + esc[i]
    return esc


def inject(n, k, pattern, rng):
    """boolean mask of length n + k marking injected positions."""
    mask = np.zeros(n + k, bool)
    if pattern == "spread":
        mask[rng.choice(n + k, k, replace=False)] = True
    else:
        a = int(0.3 * n); mask[a:a + k] = True
    return mask


def main():
    P = Paths(False); T = json.load(open(OUT / "throughput_v4.json"))
    cost = {"ciciomt_category": ("ciciomt_category", "teacher"), "ciciomt_attack": ("ciciomt_attack", "teacher"),
            "iot23_1-1": ("iot23_capture_shift", "ensemble")}
    rows = []
    for st, seed, (y, pe, ps, Hv, Ht) in streams(P):
        ck, ex = cost[st]; ms, mt = T[ck]["256"]["student_fp32"]["throughput"], T[ck]["256"][ex]["throughput"]
        ratio = lambda e: (1 / ms + 0.10 / mt) / (1 / ms + e / mt)
        tv = np.quantile(Hv, 0.9); n = len(y); top = np.flatnonzero(Ht >= np.quantile(Ht, 0.9))
        ref = m85.mf1(y, np.where(m85.online(Ht, tv), pe, ps)); ref_q = quota(Ht, tv)
        ref_qf1, ref_qblk = m85.mf1(y, np.where(ref_q, pe, ps)), max(m85.blocks(ref_q))
        for rho in RHOS:
            for pattern in ("spread", "burst"):
                rng = np.random.default_rng(1000 * seed + int(100 * rho) + (pattern == "burst"))
                k = int(round(rho * n)); src = rng.choice(top, k, replace=True); mask = inject(n, k, pattern, rng)
                H = np.empty(n + k); H[~mask] = Ht; H[mask] = Ht[src]
                e_nc = H > tv; e_on = m85.online(H, tv); e_q = quota(H, tv); g = ~mask
                bl_nc, bl_on, bl_q = m85.blocks(e_nc), m85.blocks(e_on), m85.blocks(e_q)
                rows.append({"setting": st, "seed": seed, "rho": rho, "pattern": pattern, "n_genuine": n, "n_injected": k,
                             "e_nocap": float(e_nc.mean()), "e_online": float(e_on.mean()),
                             "block_max_nocap": max(bl_nc), "block_max_online": max(bl_on),
                             "block_p95_online": float(np.quantile(bl_on, .95)),
                             "ratio_worst_nocap": ratio(max(bl_nc)), "ratio_worst_online": ratio(max(bl_on)),
                             "e_quota": float(e_q.mean()), "block_max_quota": max(bl_q), "ratio_worst_quota": ratio(max(bl_q)),
                             "genuine_escalated_quota": float(e_q[g].mean()), "f1_genuine_quota": m85.mf1(y, np.where(e_q[g], pe, ps)),
                             "genuine_escalated_nocap": float(e_nc[g].mean()), "genuine_escalated_online": float(e_on[g].mean()),
                             "f1_genuine_nocap": m85.mf1(y, np.where(e_nc[g], pe, ps)),
                             "f1_genuine_online": m85.mf1(y, np.where(e_on[g], pe, ps)),
                             "f1_student": m85.mf1(y, ps), "f1_no_attack_online": ref,
                             "f1_no_attack_quota": ref_qf1, "block_max_no_attack_quota": ref_qblk})
                r = rows[-1]
                print(st, seed, rho, pattern, f"e nc {r['e_nocap']:.3f} on {r['e_online']:.3f} blk {r['block_max_online']:.3f} "
                      f"F1 nc {r['f1_genuine_nocap']:.2f} on {r['f1_genuine_online']:.2f} stud {r['f1_student']:.2f} ref {ref:.2f}", flush=True)
    summ = {}
    for st in sorted({r["setting"] for r in rows}):
        for rho in RHOS:
            for pattern in ("spread", "burst"):
                R = [r for r in rows if r["setting"] == st and r["rho"] == rho and r["pattern"] == pattern]
                summ[f"{st}|{rho}|{pattern}"] = {k: float(np.median([r[k] for r in R])) for k in R[0]
                                                 if isinstance(R[0][k], float)}
    json.dump({"rows": rows, "summary": summ}, open(OUT / "escalation_attack.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
