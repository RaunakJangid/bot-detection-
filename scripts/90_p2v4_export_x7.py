"""P2 v4, X7 export: replay bundles for the Mininet v2 emulation (x7_mininet.py, numpy only in WSL).

Two shifted streams in stream order (as scripts/85): CICIoMT2024 6-class (student shield_s0, expert teacher_s0; fixed
random order, first 40,000 test flows) and IoT-23 1-1 (student small_mlp_kd_s0, expert ensemble_s0; time order, all
test flows of the capture). Per flow: label, student prediction, student entropy, expert prediction. Also the
validation threshold for the 10% budget and the measured batch-1 student time (outputs_v5/p2/throughput_v4.json).
Output: outputs_v5/p2/x7/x7_<setting>.npz."""
import json
from pathlib import Path

import numpy as np

from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths

OUT = Path("D:/shield_run/outputs_v5/p2/x7")
ent = lambda P: -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1)


def main():
    P = Paths(False); OUT.mkdir(parents=True, exist_ok=True); T = json.load(open(OUT.parent / "throughput_v4.json"))
    d = load_processed(P.processed("ciciomt"), "category", splits=("validation", "test")); y = d.splits["test"].y
    lc = P.v3 / "logits" / "ciciomt_category"
    o = np.random.default_rng(0).permutation(len(y))[:40_000]
    Pt = softmax(np.load(lc / "shield_s0_test.npy").astype(np.float64))[o]
    Pv = softmax(np.load(lc / "shield_s0_validation.npy").astype(np.float64))
    pe = softmax(np.load(lc / "teacher_s0_test.npy").astype(np.float64)).argmax(1)[o]
    np.savez(OUT / "x7_ciciomt6.npz", y=y[o].astype(np.int16), ps=Pt.argmax(1).astype(np.int16), H=ent(Pt).astype(np.float32),
             pe=pe.astype(np.int16), thr=np.quantile(ent(Pv), 0.9), student_s=1 / T["ciciomt_category"]["1"]["student_fp32"]["throughput"])
    Z = np.load(P.v3 / "iot23" / "scores_capture_shift.npz", allow_pickle=True)
    models = list(Z["models"]); m = Z["capture"] == "CTU-IoT-Malware-Capture-1-1"; o = np.argsort(Z["ts"][m], kind="stable")
    i = models.index("small_mlp_kd_s0"); sv, st = Z["sv"][i], Z["st"][i][m][o]
    pe = (Z["st"][models.index("ensemble_s0")][m][o] > 0.5).astype(np.int16)
    np.savez(OUT / "x7_iot23_1-1.npz", y=Z["yt"][m][o].astype(np.int16), ps=(st > 0.5).astype(np.int16),
             H=ent(np.c_[1 - st, st]).astype(np.float32), pe=pe, thr=np.quantile(ent(np.c_[1 - sv, sv]), 0.9),
             student_s=1 / T["iot23_capture_shift"]["1"]["student_fp32"]["throughput"])
    for f in sorted(OUT.glob("*.npz")):
        B = np.load(f); print(f.name, len(B["y"]), "thr", float(B["thr"]), "val-threshold share", float((B["H"] > B["thr"]).mean()))


if __name__ == "__main__":
    main()
