"""Export the live detect-and-block detector (for scripts/e7_liveblock.py, numpy only in WSL).

Small MLP (64-64, the SHIELD student size class) trained on the IoT-23 Zeek conn.log sample (all captures; 15% random
validation slice) with the Zeek features of scripts/41_iot23_shift.py minus the `service` one-hots (Zeek 8 assigns
services differently from IoT-23's Bro; see scripts/48_unsw_realhome.py). Exported: weights, z-score parameters, feature
names, and the lab threshold = 99th percentile of validation benign scores (1% false-alarm target)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("export_liveblock")
_s = importlib.util.spec_from_file_location("i23", Path(__file__).with_name("41_iot23_shift.py"))
i23 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i23)


def main():
    parser(__doc__, datasets=False).parse_args()
    P = Paths(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    A, names = i23.featurize(df)
    keep = [j for j, n in enumerate(names) if not n.startswith("svc_")]
    A, names = np.nan_to_num(A[:, keep]).astype(np.float32), [names[j] for j in keep]
    y = df.y.to_numpy()
    v = np.random.default_rng(0).random(len(y)) < 0.15
    mu, sd = A[~v].mean(0), A[~v].std(0) + 1e-6
    Z = (A - mu) / sd
    m = i23.train_mlp(Z[~v], y[~v], Z[v], y[v], (64, 64), 0, dev)
    sv = i23.mlp_score(m, Z[v], dev)
    thr = float(np.quantile(sv[y[v] == 0], 0.99))
    log.info("validation AUC %.4f | lab threshold %.4f | recall at it %.3f", roc_auc_score(y[v], sv), thr,
             (sv[y[v] == 1] > thr).mean())
    sdict = {k: t.detach().cpu().numpy() for k, t in m.state_dict().items()}
    lin = sorted({k.rsplit(".", 1)[0] for k in sdict}, key=lambda s: int(s.split(".")[1]))
    out = P.v3 / "liveblock"; out.mkdir(parents=True, exist_ok=True)
    arrays = {"mu": mu, "sd": sd, "thr": np.array(thr), "names": np.array(names), "n_layers": np.array(len(lin))}
    for i, k in enumerate(lin):
        arrays[f"W{i}"] = sdict[f"{k}.weight"].T.astype(np.float32)
        arrays[f"b{i}"] = sdict[f"{k}.bias"].astype(np.float32)
    np.savez(out / "detector.npz", **arrays)
    log.info("saved %s (%d features, %d layers)", out / "detector.npz", len(names), len(lin))


if __name__ == "__main__":
    main()
