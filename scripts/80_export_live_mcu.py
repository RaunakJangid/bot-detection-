"""Export the P1 live-testbed detector (outputs_v3/liveblock/detector.npz: 44 Zeek inputs, 64-64 MLP, 7,170 float
parameters) for the nRF52832 test, as a third model next to the int8 students of scripts/65.
Test flows are stored as exact float32: the lab threshold of the detector coincides with a score shared by about 29% of
the normal IoT-23 validation flows (identical feature vectors), so float16 rounding of the inputs would move those flows
across the threshold. The commissioning window (1,000 real-home flows) is stored as float16 to fit in flash.

The detector runs in float32 on the Cortex-M4F FPU, unchanged. Embedded data (features stored as IEEE float16 to fit in
flash; the PC reference uses exactly the same float16-rounded inputs, so device and PC see identical numbers):
  comm   the P1 real-home commissioning window: the first 1,000 flows of day 1 (UNSW smart home, time order)
  test   100 real-home flows of day 2 (every k-th flow, benign) + 50 benign and 50 malicious IoT-23 validation flows
         (the 15% random validation slice of scripts/61, seed 0)
PC reference (numpy float32, same accumulation order as the C code: acc = b[j]; acc += w[j][i] * x[i] for i = 0..):
  predicted class and attack score (softmax, class 1) per test flow; Step 1 threshold = split-conformal quantile of
  Eq. (7) on the 1,000 commissioning scores (k = ceil(1,001 x 0.99) = 991); KS distance between the commissioning scores
  and the 100 real-home test scores; the lab threshold of the detector.
Output: D:/shield_run/mcu/export/model_live.h and export_live_summary.json."""

import glob
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from shield.utils.config import Paths

OUT = Path("D:/shield_run/mcu/export")
N_COMM, N_HOME, N_IOT = 1000, 100, 50
_s = importlib.util.spec_from_file_location("u48", Path(__file__).with_name("48_unsw_realhome.py"))
u48 = importlib.util.module_from_spec(_s); _s.loader.exec_module(u48)
i23 = u48.i23


def forward_c_order(Ws, bs, x):
    """float32 forward pass with the C loop order (sequential accumulation per output)."""
    h = x.astype(np.float32)
    for li, (W, b) in enumerate(zip(Ws, bs)):
        Wt = W.T.astype(np.float32)                       # [out][in]
        acc = np.repeat(b.astype(np.float32)[None, :], len(h), 0)
        for i in range(Wt.shape[1]):
            acc = (acc + Wt[:, i][None, :] * h[:, i][:, None]).astype(np.float32)
        h = np.maximum(acc, np.float32(0)) if li < len(Ws) - 1 else acc
    return h


def score(logits):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z.astype(np.float32)).astype(np.float32)
    return (e[:, 1] / e.sum(1)).astype(np.float32)


def carr(name, ctype, a, per_line=12):
    flat = np.asarray(a).ravel()
    if ctype == "float":
        items = [f"{float(v):.9e}f" for v in flat]
    else:
        items = [str(int(v)) for v in flat]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return f"static const {ctype} {name}[{len(flat)}] = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def main():
    P = Paths(False)
    B = np.load(P.v3 / "liveblock" / "detector.npz")
    nl = int(B["n_layers"]); Ws = [B[f"W{i}"] for i in range(nl)]; bs = [B[f"b{i}"] for i in range(nl)]
    names = [str(n) for n in B["names"]]; mu, sd, thr = B["mu"], B["sd"], float(B["thr"])
    assert [W.shape for W in Ws] == [(44, 64), (64, 64), (64, 2)], [W.shape for W in Ws]

    def feats(df):
        A, nm = i23.featurize(df)
        idx = [nm.index(n) for n in names]
        return ((np.nan_to_num(A[:, idx]) - mu) / sd).astype(np.float32)

    days = sorted(glob.glob(str(P.v3 / "realbenign" / "zeek_unsw" / "conn_*.log")))
    d1, d2 = (u48.read_zeek(f) for f in days[:2])
    d1 = d1.iloc[np.argsort(d1.ts.to_numpy(), kind="stable")].reset_index(drop=True)
    d2 = d2.iloc[np.argsort(d2.ts.to_numpy(), kind="stable")].reset_index(drop=True)
    x_comm = feats(d1.iloc[:N_COMM])
    step = len(d2) // N_HOME
    x_home = feats(d2.iloc[::step].iloc[:N_HOME])
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet")
    v = np.random.default_rng(0).random(len(df)) < 0.15                      # validation slice of scripts/61
    vb, vm = np.flatnonzero(v & (df.y.to_numpy() == 0)), np.flatnonzero(v & (df.y.to_numpy() == 1))
    g = np.random.default_rng(80)
    pick = np.r_[np.sort(g.choice(vb, N_IOT, replace=False)), np.sort(g.choice(vm, N_IOT, replace=False))]
    x_iot = feats(df.iloc[pick].reset_index(drop=True))
    x_test = np.vstack([x_home, x_iot]); y_test = np.r_[np.zeros(N_HOME + N_IOT, np.uint8), np.ones(N_IOT, np.uint8)]
    src = np.r_[np.zeros(N_HOME, np.uint8), np.ones(2 * N_IOT, np.uint8)]  # 0 = real home, 1 = IoT-23

    # stored form: real-home flows (commissioning window and home test flows) as float16, so that the KS check compares
    # two samples with the same input rounding; IoT-23 test flows as exact float32 (tied scores at the lab threshold)
    h_comm, h_home = x_comm.astype(np.float16), x_home.astype(np.float16)
    e_comm = h_comm.astype(np.float32)                                        # what the device computes on
    e_test = np.vstack([h_home.astype(np.float32), x_iot.astype(np.float32)])
    l_comm, l_test = forward_c_order(Ws, bs, e_comm), forward_c_order(Ws, bs, e_test)
    s_comm, s_test = score(l_comm), score(l_test); pred = l_test.argmax(1).astype(np.uint8)
    k = int(math.ceil((N_COMM + 1) * 0.99)); t_c = float(np.sort(s_comm)[k - 1])
    a, b = np.sort(s_comm), np.sort(s_test[:N_HOME]); grid = np.r_[a, b]
    ks = float(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)).max())
    # agreement of the float16-input reference with the float32-input detector (the paper's numbers)
    s_test32 = score(forward_c_order(Ws, bs, x_test)); s_comm32 = score(forward_c_order(Ws, bs, x_comm))
    t_c32 = float(np.sort(s_comm32)[k - 1])

    h = ["/* live detector of P1 (44 Zeek inputs, 64-64, 7,170 float32 parameters); generated by scripts/80_export_live_mcu.py */\n",
         f"#define LIVE_N_COMM {N_COMM}\n#define LIVE_N_TEST {len(x_test)}\n#define LIVE_N_HOME {N_HOME}\n#define LIVE_K_INDEX {k}\n",
         f"#define LIVE_REF_THRESHOLD {t_c:.9e}f\n#define LIVE_REF_KS {ks:.9e}f\n#define LIVE_LAB_THRESHOLD {thr:.9e}f\n"]
    for i, (W, bb) in enumerate(zip(Ws, bs)):
        h.append(carr(f"live_w{i}", "float", W.T.astype(np.float32)))
        h.append(carr(f"live_b{i}", "float", bb.astype(np.float32)))
    h.append(carr("live_x_comm_f16", "uint16_t", h_comm.view(np.uint16), 16))
    h.append(carr("live_x_home_f16", "uint16_t", h_home.view(np.uint16), 16))
    h.append(carr("live_x_iot_f32", "float", e_test[N_HOME:]))
    h.append(carr("live_y_test", "uint8_t", y_test, 32))
    h.append(carr("live_src", "uint8_t", src, 32))
    h.append(carr("live_ref_pred", "uint8_t", pred, 32))
    h.append(carr("live_ref_score", "float", s_test))
    (OUT / "model_live.h").write_text("".join(h))
    summ = {"n_comm": N_COMM, "n_test": int(len(x_test)), "k_index": k, "ref_threshold_f16_inputs": t_c,
            "ref_threshold_f32_inputs": t_c32, "ref_ks_comm_vs_home": ks, "lab_threshold": thr,
            "f16_vs_f32_max_score_diff_test": float(np.abs(s_test - s_test32).max()),
            "f16_vs_f32_max_score_diff_comm": float(np.abs(s_comm - s_comm32).max()),
            "pred_attack_share": {"home": float(pred[:N_HOME].mean()), "iot_benign": float(pred[N_HOME:N_HOME + N_IOT].mean()),
                                  "iot_attack": float(pred[N_HOME + N_IOT:].mean())},
            "flagged_at_lab_iot_normal": float((s_test[N_HOME:N_HOME + N_IOT] > thr).mean()),
            "flagged_at_lab": {"home": float((s_test[:N_HOME] > thr).mean()), "iot_attack": float((s_test[N_HOME + N_IOT:] > thr).mean())},
            "flagged_at_step1": {"home": float((s_test[:N_HOME] > t_c).mean()), "iot_attack": float((s_test[N_HOME + N_IOT:] > t_c).mean())},
            "bytes": {"weights": int(4 * sum(W.size + bb.size for W, bb in zip(Ws, bs))), "comm": int(h_comm.nbytes), "test": int(h_home.nbytes + 4 * e_test[N_HOME:].size)}}
    json.dump(summ, open(OUT / "export_live_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
