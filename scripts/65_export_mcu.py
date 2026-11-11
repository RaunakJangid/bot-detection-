"""Export integer-only SHIELD students for the nRF52832 (nRF52 DK) on-device test.

For each exported model the int8 model of 13_int8.py is rebuilt exactly (same student, same seeded calibration
rows, same calibration choice) and checked against the recorded test macro-F1. The hidden-layer requantisation
(acc * acc_scale / s_act, done in float64 by IntMLP) is replaced by the standard integer form used on MCUs:
an int32 multiplier M0 and a right shift per output channel, with round-half-up. The integer reference
implemented here in numpy is what the C code computes bit for bit; its agreement with IntMLP is reported.

Written to <out>/ (default D:/shield_run/mcu/export):
  model_<name>.h   weights, biases, multipliers, shifts, output scales, input scales, test vectors, references
  export_summary.json   per model: sizes, F1 (IntMLP vs integer reference), agreement, reference timings inputs
Test vectors per model: 1,000 test flows (seeded random), 1,000 benign validation flows (commissioning window).
"""

import json
from pathlib import Path

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.predictions import load_student, run_dir
from shield.models.quant import IntMLP, calibrate
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("export_mcu")
# name -> (dataset, task, variant, seed); the calibration is read from outputs_v3/int8 (must be PTQ)
MODELS = {"ciciot34": ("ciciot", "class34", "shield", 1), "ciciomt_bin": ("ciciomt", "binary", "shield", 1)}
N_TEST, N_COMM, ALPHA = 1000, 1000, 0.01


def balanced_sample(y, per_class, rng):
    idx = [rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


def fixed_point(m):
    """m > 0 -> (M0 in [2^30, 2^31), total right shift s) with m ~= M0 / 2^s."""
    f, e = np.frexp(m)
    M0 = np.round(f * 2.0 ** 31).astype(np.int64)
    over = M0 == 2 ** 31
    M0[over] //= 2
    e = e + over
    s = 31 - e
    assert (s > 0).all() and (s < 63).all(), "shift out of range"
    return M0, s.astype(np.int64)


def int_forward(q, xq, mults):
    """Integer reference: exactly what shield_mlp.c computes. Returns float32 logits."""
    h = xq.astype(np.int64)
    for j, (Wq, bq, acc_scale) in enumerate(q.layers):
        acc = h @ Wq.T.astype(np.int64) + bq
        assert np.abs(acc).max() < 2 ** 31, "int32 accumulator overflow"
        if j == len(q.layers) - 1:
            return acc.astype(np.float32) * acc_scale.astype(np.float32)
        M0, s = mults[j]
        a = np.maximum(acc, 0)
        y = (a * M0 + (np.int64(1) << (s - 1))) >> s
        h = np.minimum(y, 255)


def softmax_score(logits, benign):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z.astype(np.float32))
    return (1.0 - p[:, benign] / p.sum(1)).astype(np.float32)


def carr(name, ctype, a, per_line=16):
    a = np.asarray(a).ravel()
    fmt = (lambda v: f"{v:.9e}f") if ctype == "float" else str
    body = ",\n  ".join(", ".join(fmt(v) for v in a[i:i + per_line]) for i in range(0, len(a), per_line))
    return f"static const {ctype} {name}[{len(a)}] = {{\n  {body}\n}};\n"


def main():
    p = parser(__doc__)
    p.add_argument("--out", default="D:/shield_run/mcu/export")
    args = p.parse_args()
    paths = Paths(False)
    tcfg = load_config("teacher", False)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (ds, task, variant, seed) in MODELS.items():
        rec = load_json(paths.v3 / "int8" / f"{ds}_{task}" / f"{variant}_s{seed}.json")["row"]
        assert rec["mode"] == "ptq", f"{name}: only PTQ models can be rebuilt exactly"
        cset, ip, ap = rec["calibration_w8a8"].split("/")
        data = load_processed(paths.processed(ds), task)
        tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
        model, S = load_student(run_dir(paths, ds, task, variant, seed, tcfg["model"]), "cpu")
        model = model.cpu()
        cols = lambda X: X if len(S) == X.shape[1] else np.asarray(X[:, S])
        rng = np.random.default_rng(seed)                      # same draw order as 13_int8.py
        prop = cols(np.asarray(tr.X[stratified_sample(tr.y, 200_000, rng, 50)]))
        cal_rows = prop if cset == "prop" else cols(np.asarray(tr.X[balanced_sample(tr.y, 200_000 // data.n_classes, rng)]))
        q = IntMLP(model, calibrate(model, cal_rows, float(ip), float(ap)), act_bits=8)
        Xt, Xv = cols(te.X), cols(va.X)
        f1 = lambda y, z: classification_metrics(y, z, data.n_classes)["macro_f1"]
        lt = q.forward(Xt)
        f1_q = f1(te.y, lt)
        assert abs(f1_q - rec["test_int8"]) < 1e-9, f"{name}: rebuilt model differs ({f1_q} vs {rec['test_int8']})"
        mults = [fixed_point(acc_scale / s) for (_, _, acc_scale), s in zip(q.layers[:-1], q.s_act)]
        li = int_forward(q, q.quantize_input(Xt), mults)
        agree = float((li.argmax(1) == lt.argmax(1)).mean())
        f1_i = f1(te.y, li)
        benign = [i for i, c in enumerate(data.classes) if "benign" in str(c).lower()][0]
        log.info("%s: IntMLP test F1 %.4f (recorded %.4f); integer reference %.4f, prediction agreement %.6f",
                 name, f1_q, rec["test_int8"], f1_i, agree)

        g = np.random.default_rng(0)
        ti = np.sort(g.choice(len(te.y), N_TEST, replace=False))
        bi = np.sort(g.choice(np.flatnonzero(va.y == benign), N_COMM, replace=False))
        x_test, x_comm = np.asarray(Xt[ti], np.float32), np.asarray(Xv[bi], np.float32)
        xq_test, xq_comm = q.quantize_input(x_test), q.quantize_input(x_comm)
        l_test, l_comm = int_forward(q, xq_test, mults), int_forward(q, xq_comm, mults)
        pred = l_test.argmax(1).astype(np.uint8)
        s_comm, s_test = softmax_score(l_comm, benign), softmax_score(l_test, benign)
        k = int(np.ceil((N_COMM + 1) * (1 - ALPHA)))            # split-conformal index (1-based)
        t_c = float(np.sort(s_comm)[k - 1])
        # KS distance between the commissioning and test score samples
        a, b = np.sort(s_comm), np.sort(s_test)
        grid = np.concatenate([a, b])
        ks = float(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)).max())

        dims = [Wq.shape[1] for Wq, _, _ in q.layers] + [q.layers[-1][0].shape[0]]
        P = name.upper()
        h = [f"/* {name}: {ds} {task} {variant} seed {seed}; generated by scripts/65_export_mcu.py */\n",
             f"#define {P}_N_LAYERS {len(q.layers)}\n",
             f"#define {P}_DIMS {{{', '.join(map(str, dims))}}}\n",
             f"#define {P}_BENIGN {benign}\n#define {P}_N_TEST {N_TEST}\n#define {P}_N_COMM {N_COMM}\n",
             f"#define {P}_REF_THRESHOLD {t_c:.9e}f\n#define {P}_REF_KS {ks:.9e}f\n#define {P}_K_INDEX {k}\n"]
        for j, (Wq, bq, acc_scale) in enumerate(q.layers):
            h.append(carr(f"{name}_w{j}", "int8_t", Wq))
            h.append(carr(f"{name}_b{j}", "int32_t", bq.astype(np.int64)))
            if j < len(q.layers) - 1:
                h.append(carr(f"{name}_m{j}", "int32_t", mults[j][0]))
                h.append(carr(f"{name}_s{j}", "uint8_t", mults[j][1]))
            else:
                h.append(carr(f"{name}_out_scale", "float", acc_scale.astype(np.float32)))
        h.append(carr(f"{name}_inv_s_in", "float", (1.0 / q.s_in).astype(np.float32)))
        h.append(carr(f"{name}_x_test_f32", "float", x_test, 8))
        h.append(carr(f"{name}_x_test_q", "int8_t", xq_test, 32))
        h.append(carr(f"{name}_x_comm_q", "int8_t", xq_comm, 32))
        h.append(carr(f"{name}_ref_pred", "uint8_t", pred, 32))
        h.append(carr(f"{name}_ref_score", "float", s_test, 8))
        h.append(carr(f"{name}_y_test", "uint8_t", te.y[ti].astype(np.uint8), 32))
        (out / f"model_{name}.h").write_text("#include <stdint.h>\n" + "".join(h))
        params = int(sum(W.size + b.size for W, b, _ in q.layers))
        summary[name] = {"dataset": ds, "task": task, "seed": seed, "calibration": rec["calibration_w8a8"],
                         "dims": dims, "params": params, "int8_bytes_recorded": rec["int8_bytes"],
                         "weights_bytes": int(sum(W.size for W, _, _ in q.layers)),
                         "test_f1_intmlp": f1_q, "test_f1_integer_ref": f1_i, "pred_agreement_full_test": agree,
                         "n_test_full": int(len(te.y)), "benign_class": int(benign),
                         "ref_threshold_k1000": t_c, "ref_ks": ks,
                         "subset_f1_integer_ref": f1(te.y[ti], l_test)}
        log.info("%s: wrote model_%s.h (%d params, dims %s)", name, name, params, dims)
    save_json(summary, out / "export_summary.json")
    log.info("summary: %s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
