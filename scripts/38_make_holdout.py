"""E5: fresh holdout for the final CICIoT2023 pipeline (the old test set was viewed during development).

Builds processed dataset `ciciot_fullho` from `ciciot_full` without touching the raw data:
  * holdout = a uniform random sample of N rows of the full TRAINING pool (rows no model selection ever scored);
    these rows become the new `test` split, scored once at the end of the chain;
  * train = the remaining training rows; validation = the unchanged strict validation rows (all choices are made
    there exactly as before, by the same frozen procedure);
  * the full training pool is already de-duplicated internally, so no holdout vector occurs in the new train;
  * the scaler is re-fitted on the new train only: features are signed-log then z-scored, so refitting is an exact
    affine re-standardisation of the stored arrays (X' = (X - m) / s with m, s the new-train mean/std).
The holdout has the natural class mix of the full data (not the capped strict test), so macro-F1 is reported on
it alongside per-class F1."""

import json

import numpy as np

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("holdout")
TASKS = ("binary", "family", "class34")


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--n", type=int, default=2_000_000)
    args = p.parse_args()
    P = Paths(args.smoke)
    src, dst = P.processed("ciciot_full"), P.processed("ciciot_fullho")
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(src / "meta.json"))
    Xtr = np.load(src / "X_train.npy", mmap_mode="r")
    n = len(Xtr)
    rng = np.random.default_rng(20261011)
    ho = np.zeros(n, bool); ho[rng.choice(n, args.n, replace=False)] = True
    tr_idx, ho_idx = np.flatnonzero(~ho), np.flatnonzero(ho)

    # new-train statistics in the stored (already standardised) space, in chunks
    s1 = np.zeros(Xtr.shape[1]); s2 = np.zeros(Xtr.shape[1])
    for i in range(0, len(tr_idx), 2_000_000):
        b = np.asarray(Xtr[tr_idx[i:i + 2_000_000]], np.float64)
        s1 += b.sum(0); s2 += (b * b).sum(0)
    m = s1 / len(tr_idx); s = np.sqrt(np.maximum(s2 / len(tr_idx) - m * m, 1e-12))
    s[s < 1e-6] = 1.0
    restd = lambda A: ((np.asarray(A, np.float64) - m) / s).astype(np.float32)

    out = np.lib.format.open_memmap(dst / "X_train.npy", mode="w+", dtype=np.float32, shape=(len(tr_idx), Xtr.shape[1]))
    for i in range(0, len(tr_idx), 2_000_000):
        out[i:i + 2_000_000] = restd(Xtr[tr_idx[i:i + 2_000_000]])
    out.flush(); del out
    np.save(dst / "X_test.npy", restd(Xtr[ho_idx]))
    np.save(dst / "X_validation.npy", restd(np.load(src / "X_validation.npy")))
    counts = {}
    for t in TASKS:
        y = np.load(src / f"y_{t}_train.npy")
        np.save(dst / f"y_{t}_train.npy", y[tr_idx]); np.save(dst / f"y_{t}_test.npy", y[ho_idx])
        np.save(dst / f"y_{t}_validation.npy", np.load(src / f"y_{t}_validation.npy"))
        counts[t] = {"train": np.bincount(y[tr_idx], minlength=len(meta["tasks"][t])).tolist(),
                     "test": np.bincount(y[ho_idx], minlength=len(meta["tasks"][t])).tolist()}
    # compose the scaler: stored z = (log - mean) / std ; new z' = (z - m) / s = (log - (mean + m*std)) / (std*s)
    sc = meta["scaler"]
    mean, std = np.asarray(sc["mean"]), np.asarray(sc["std"])
    sc["mean"] = (mean + m * std).tolist(); sc["std"] = (std * s).tolist()
    meta.update({"dataset": "ciciot_fullho", "scaler": sc,
                 "n_rows": {"train": int(len(tr_idx)), "validation": int(meta["n_rows"]["validation"]), "test": int(len(ho_idx))},
                 "holdout": {"from": "ciciot_full train pool", "n": int(len(ho_idx)), "seed": 20261011,
                             "scaler": "refit on new train (affine re-standardisation)"}})
    meta["class_counts_holdout"] = counts
    save_json(meta, dst / "meta.json")
    log.info("ciciot_fullho: train %d, validation %d, holdout(test) %d", len(tr_idx), meta["n_rows"]["validation"], len(ho_idx))


if __name__ == "__main__":
    main()
