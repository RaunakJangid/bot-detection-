import numpy as np
import torch
from shield.data.datasets import load_processed, predict_logits, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.predictions import load_student, run_dir
from shield.models.quant import calibrate, _fq, _linears
from shield.utils.config import Paths

paths = Paths()
data = load_processed(paths.processed("ciciot"), "family")
tr, va = data.splits["train"], data.splits["validation"]
model, S = load_student(run_dir(paths, "ciciot", "family", "shield", 0), torch.device("cpu"))
Xv = torch.as_tensor(np.asarray(va.X[:, S]))
Xcal = np.asarray(tr.X[stratified_sample(tr.y, 200000, np.random.default_rng(0), 50)])[:, S]
f1 = lambda z: classification_metrics(va.y, z.detach().numpy(), data.n_classes)["macro_f1"]


def run(q_in, q_w, q_act, pct=99.99, bits_in=8):
    cal = calibrate(model, Xcal, pct)
    lim = 2 ** (bits_in - 1) - 1
    s_in = torch.as_tensor(cal["in_max"] / lim, dtype=torch.float32)
    h = _fq(Xv, s_in, -lim, lim) if q_in else Xv
    lins = _linears(model)
    for i, lin in enumerate(lins):
        w = lin.weight
        if q_w:
            w = _fq(w, (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12), -127, 127)
        h = torch.nn.functional.linear(h, w, lin.bias)
        if i < len(lins) - 1:
            h = torch.relu(h)
            if q_act:
                h = _fq(h, torch.tensor(cal["act_max"][i] / 255), 0, 255)
    return f1(h)


def run2(compand, act_pc, pct=99.99, a=4.0):
    """compand: 8-bit code on asinh(a*x)/a grid, decoded by LUT; act_pc: per-channel activation scales."""
    X = Xv.numpy().astype(np.float64)
    if compand:
        u = np.arcsinh(a * Xcal) / a
        umax = np.maximum(np.percentile(np.abs(u), 100, axis=0), 1e-6) / 127
        c = np.clip(np.round(np.arcsinh(a * X) / a / umax), -127, 127)
        X = np.sinh(c * umax * a) / a
    else:
        s = np.maximum(np.percentile(np.abs(Xcal), 100, axis=0), 1e-6) / 127
        X = np.clip(np.round(X / s), -127, 127) * s
    h = torch.as_tensor(X, dtype=torch.float32)
    hc = torch.as_tensor(Xcal, dtype=torch.float32)
    lins = _linears(model)
    for i, lin in enumerate(lins):
        h, hc = lin(h), lin(hc)
        if i < len(lins) - 1:
            h, hc = torch.relu(h), torch.relu(hc)
            amax = (torch.quantile(hc[:50000], pct / 100, dim=0) if act_pc
                    else torch.quantile(hc[:50000].flatten()[:5_000_000], pct / 100)).clamp(min=1e-6)
            h = _fq(h, amax / 255, 0, 255)
    return f1(h)


with torch.no_grad():
    for comp in (0, 1):
        for pc in (0, 1):
            print(f"compand={comp} act_per_channel={pc}: val F1 {run2(comp, pc):.4f}")
    for a in (1.0, 16.0):
        print(f"compand a={a} per-channel: {run2(1, 1, a=a):.4f}")

    print("fp32", f1(model(Xv)))
    print("input only", run(1, 0, 0), " weights only", run(0, 1, 0), " act only", run(0, 0, 1))
    for pct in (99.9, 99.99, 100):
        print("input only pct", pct, run(1, 0, 0, pct))
    print("input int16", run(1, 0, 0, 100, 16))
    q = np.percentile(np.abs(Xcal), [50, 99, 99.99, 100], axis=0)
    print("per-feature |x| p50 / p99 / p99.99 / max (first 12):\n", np.round(q[:, :12], 3))
