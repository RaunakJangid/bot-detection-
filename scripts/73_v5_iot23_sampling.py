"""v5 A3: what the IoT-23 sampler of scripts/41 (read_archive, cap 300,000 per label per capture) did to each capture.
From outputs_v3/iot23/sample.parquet: kept flows per capture and label, flows seen in the archive (kept x w, w = seen/kept),
true vs sampled attack share, and whether the benign stream was kept complete (no thinning, so time order is intact).
Output: outputs_v5/iot23_sampling_v5.json."""
import json
import numpy as np
import pandas as pd
df = pd.read_parquet("D:/shield_run/outputs_v3/iot23/sample.parquet", columns=["capture", "y", "w"])
out = {}
for c, g in df.groupby("capture"):
    kb, km = int((g.y == 0).sum()), int((g.y == 1).sum())
    wb = float(g.w[g.y == 0].iloc[0]) if kb else 0.0; wm = float(g.w[g.y == 1].iloc[0]) if km else 0.0
    sb, sm = kb * wb, km * wm
    out[c] = {"kept_benign": kb, "kept_malicious": km, "seen_benign": round(sb), "seen_malicious": round(sm),
              "benign_complete": bool(abs(wb - 1) < 1e-9) if kb else None, "malicious_complete": bool(abs(wm - 1) < 1e-9) if km else None,
              "sample_attack_share": km / max(kb + km, 1), "true_attack_share": sm / max(sb + sm, 1)}
json.dump(out, open("D:/shield_run/outputs_v5/iot23_sampling_v5.json", "w"), indent=1)
for c, v in out.items():
    print(f"{c:36s} kept b/m {v['kept_benign']:7d}/{v['kept_malicious']:7d} seen b/m {v['seen_benign']:9d}/{v['seen_malicious']:10d} "
          f"benign complete {v['benign_complete']} mal complete {v['malicious_complete']} share sample {v['sample_attack_share']:.3f} true {v['true_attack_share']:.4f}")
