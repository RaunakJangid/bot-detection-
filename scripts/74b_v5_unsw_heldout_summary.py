"""v5 B2 summary: for the real-home models retrained on the training captures of the same-family and capture-shift designs
(scripts/74), per model: lab threshold t0 (99th percentile of IoT-23 validation benign scores) and Step 1 threshold tc
(split-conformal on the first 1,000 real-home flows of day 1); real-home FPR at t0 (days 1 to 3) and at tc (remaining
flows); recall at t0 and tc on IoT-23 validation attacks (as in v4) and on the attacks of the held-out captures.
Output: outputs_v5/unsw_heldout_summary.json."""
import json, math
import numpy as np

def conformal(s, a=0.01):
    k = int(math.ceil((len(s) + 1) * (1 - a))); return np.sort(s)[k - 1]

out = {}
for design in ("same_family", "capture_shift"):
    Z = np.load(f"D:/shield_run/outputs_v5/regime/unsw_{design}.npz", allow_pickle=True)
    days = [str(d) for d in Z["days"]]; yv = Z["yv"].astype(bool); rows = []
    for i, m in enumerate(Z["models"]):
        sv = Z["sv"][i].astype(np.float64); t0 = np.quantile(sv[~yv], 0.99)
        allu = np.concatenate([Z[f"u_{d}"][i] for d in days]).astype(np.float64); tc = conformal(allu[:1000])
        ha = Z["ha"][i].astype(np.float64)
        rows.append({"model": str(m), "home_fpr_lab": float((allu > t0).mean()), "home_fpr_step1": float((allu[1000:] > tc).mean()),
                     "val_recall_lab": float((sv[yv] > t0).mean()), "val_recall_step1": float((sv[yv] > tc).mean()),
                     "heldout_recall_lab": float((ha > t0).mean()), "heldout_recall_step1": float((ha > tc).mean())})
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "model"}
    rng = {k: [float(min(r[k] for r in rows)), float(max(r[k] for r in rows))] for k in rows[0] if k != "model"}
    p95 = {k: float(np.quantile([r[k] for r in rows], .95)) for k in ("home_fpr_lab", "home_fpr_step1")}
    out[design] = {"test_captures": [str(c) for c in Z["test_caps"]], "median": med, "range": rng, "p95": p95, "per_model": rows}
    print(design, {k: round(100 * v, 1) for k, v in med.items()}, "p95", {k: round(100 * v, 2) for k, v in p95.items()})
json.dump(out, open("D:/shield_run/outputs_v5/unsw_heldout_summary.json", "w"), indent=1)
