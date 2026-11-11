"""Cycle 2 pilot (C9): attribute an escalation surge to benign distribution shift or to uncertainty flooding, without
labels. Criteria frozen in papers/novelty/RESEARCH_LOG.md (Cycle 2, Stage 1). CPU only, no training.

Per setting and student seed:
  validation entropies split at random into a reference half (threshold t_H = its 90th percentile, reference
  distribution) and an evaluation half (genuine flows of flood windows): no flow is in both;
  shift windows: consecutive 300-flow windows of the shifted test stream (stream order as scripts/85), at most 200
  per setting, kept if their escalated share is a surge (above 12%);
  flood windows, one per shift window, at the same escalated share e*: 300 genuine evaluation-half flows plus k
  injected flows.
    mimicry: k_min = n (e* - e_g) / (1 - e*) copies of reference flows above the 90th percentile;
    adaptive, budget eps in {minimal, 0.25, 0.5, 1.0} (k = max(k_min, eps n)): k_t injected above t_H to hit e*,
    drawn from the shifted stream's tail; k - k_t below t_H, drawn where the shifted stream's bulk density exceeds
    the window's own bulk density most (positive part of the histogram difference).
Statistics per window (higher = more shift-like): escalation rate; full-distribution KS / critical value; entropy PSI
(reference deciles); ours = KS on the bulk (entropies <= t_H) / critical value.
AUROC: shift windows (1) against flood windows (0), pooled over settings, per seed and attacker.
Output: papers/novelty/runs/cycle_2/pilot_c9.json."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "C:/Users/Prakash/shield/src")
OUT = "D:/shield_run/papers/novelty/runs/cycle_2"
N, MAXW, SURGE = 300, 200, 0.12
BUDGETS = ("min", 0.25, 0.5, 1.0)


def ent(P):
    P = np.clip(P, 1e-12, 1); return -(P * np.log(P)).sum(1)


def ks(a, b):
    a, b = np.sort(a), np.sort(b); v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def crit(n1, n2):
    return 1.949 * np.sqrt((n1 + n2) / (n1 * n2))


def psi(ref, cur):
    e = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(e) < 3:
        return 0.0
    pr = np.histogram(np.clip(ref, e[0], e[-1]), e)[0] / len(ref) + 1e-4
    pc = np.histogram(np.clip(cur, e[0], e[-1]), e)[0] / len(cur) + 1e-4
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def stats(H, ref, tH):
    hb, rb = H[H <= tH], ref[ref <= tH]
    return {"esc": float(np.mean(H > tH)), "full_ks": ks(H, ref) / crit(len(H), len(ref)), "psi": psi(ref, H),
            "bulk_ks": ks(hb, rb) / crit(len(hb), len(rb)) if len(hb) > 5 else 0.0}


def load(setting, seed):
    from shield.data.datasets import load_processed  # noqa: F401  (paths only)
    from shield.eval.metrics import softmax
    from shield.utils.config import Paths
    P = Paths(False)
    if setting.startswith("ciciomt"):
        lc = P.v3 / "logits" / setting
        Hv = ent(softmax(np.load(lc / f"shield_s{seed}_validation.npy").astype(np.float64)))
        Ht = ent(softmax(np.load(lc / f"shield_s{seed}_test.npy").astype(np.float64)))
        Ht = Ht[np.random.default_rng(0).permutation(len(Ht))]
        return Hv, Ht
    design, cap = setting.split("|")
    Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True)
    models = list(Z["models"]); i = models.index(f"small_mlp_kd_s{seed}"); m = Z["capture"] == cap
    sv, st = Z["sv"][i], Z["st"][i][m][np.argsort(Z["ts"][m], kind="stable")]
    return ent(np.c_[1 - sv, sv]), ent(np.c_[1 - st, st])


def job(args):
    setting, seed = args
    rng = np.random.default_rng(1000 + seed)
    Hv, Ht = load(setting, seed)
    perm = rng.permutation(len(Hv)); ref, ev = Hv[perm[: len(Hv) // 2]], Hv[perm[len(Hv) // 2:]]
    tH = float(np.quantile(ref, 0.9)); top = ref[ref > tH]
    starts = np.arange(0, len(Ht) - N + 1, N)
    if len(starts) > MAXW:
        starts = np.sort(rng.choice(starts, MAXW, replace=False))
    tail_shift, bulk_shift = Ht[Ht > tH], Ht[Ht <= tH]
    edges = np.linspace(0, tH, 31)
    rows = {"shift": [], "mimicry": [], **{f"adaptive_{b}": [] for b in BUDGETS}}
    for s0 in starts:
        W = Ht[s0:s0 + N]; e_star = float(np.mean(W > tH))
        if e_star <= SURGE:
            continue
        rows["shift"].append(stats(W, ref, tH))
        G = ev[rng.choice(len(ev), N, replace=False)]; e_g = float(np.mean(G > tH))
        k_min = max(1, int(np.ceil(N * (e_star - e_g) / (1 - e_star))))
        rows["mimicry"].append(stats(np.concatenate([G, rng.choice(top, k_min)]), ref, tH))
        for b in BUDGETS:
            k = k_min if b == "min" else max(k_min, int(b * N))
            k_t = int(np.clip(round(e_star * (N + k) - e_g * N), 0, k)); k_b = k - k_t
            inj = [rng.choice(tail_shift if len(tail_shift) else top, k_t)]
            if k_b > 0:
                gb = G[G <= tH]
                d = np.histogram(bulk_shift, edges)[0] / max(len(bulk_shift), 1) - np.histogram(gb, edges)[0] / max(len(gb), 1)
                w = np.clip(d, 0, None)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
                bins = rng.choice(len(w), k_b, p=w)
                inj.append(rng.uniform(edges[bins], edges[bins + 1]))
            rows[f"adaptive_{b}"].append(stats(np.concatenate([G] + inj), ref, tH))
    return {"setting": setting, "seed": seed, "tH": tH, "n_test": len(Ht), "n_val": len(Hv), "rows": rows,
            "e_stream_fullval": float(np.mean(Ht > np.quantile(Hv, 0.9)))}


SETTINGS = ["ciciomt_category", "ciciomt_attack", "capture_shift|CTU-IoT-Malware-Capture-1-1",
            "same_family|CTU-IoT-Malware-Capture-49-1", "same_family|CTU-IoT-Malware-Capture-7-1"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    os.makedirs(OUT, exist_ok=True)
    with Pool(min(15, len(SETTINGS) * len(seeds))) as pool:
        res = pool.map(job, [(s, k) for s in SETTINGS for k in seeds])
    out = {"jobs": [{k: v for k, v in r.items() if k != "rows"} | {"n_shift_windows": len(r["rows"]["shift"])} for r in res],
           "auroc": {}}
    for seed in seeds:
        R = [r for r in res if r["seed"] == seed]
        pos = {m: [w[m] for r in R for w in r["rows"]["shift"]] for m in ("esc", "full_ks", "psi", "bulk_ks")}
        for att in ["mimicry"] + [f"adaptive_{b}" for b in BUDGETS]:
            neg = {m: [w[m] for r in R for w in r["rows"][att]] for m in pos}
            out["auroc"][f"{seed}|{att}"] = {m: auroc(pos[m], neg[m]) for m in pos}
    json.dump(out, open(f"{OUT}/pilot_c9_s{'-'.join(map(str, seeds))}.json", "w"), indent=1)
    for k, v in out["auroc"].items():
        print(k, {m: round(x, 3) for m, x in v.items()})
    for j in out["jobs"]:
        print(j)


if __name__ == "__main__":
    main()
