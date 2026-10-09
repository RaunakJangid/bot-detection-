"""Integer-only int8 inference for StudentMLP (v3 step 4), replacing v2's dynamic quantisation.

v2's `quantize_dynamic` quantises each batch of inputs with ONE scale, so a few heavy-tailed flow
features set the range and the small informative ones round to 0. Here:
- every input feature gets its own symmetric int8 scale (calibrated percentile of |x| on train rows),
  folded into the first layer's weights, so the layer still takes one int8 vector;
- weights are int8 per output channel (symmetric), biases int32, accumulation int32;
- hidden activations (after ReLU) are uint8 with a calibrated per-layer scale; the last layer's int32
  accumulator is dequantised to logits.
`IntMLP.forward` simulates exactly that arithmetic in float64 (every value is an integer times a known
scale), so its accuracy is the accuracy of the integer model.
QAT: `qat_finetune` trains the float model through the same rounding with a straight-through estimator.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from shield.models.student import StudentMLP


def _linears(model: StudentMLP) -> list[nn.Linear]:
    return [m for m in model.net if isinstance(m, nn.Linear)]


@torch.no_grad()
def calibrate(model: StudentMLP, X: np.ndarray, pct: float = 99.99, act_pct: float | None = None) -> dict:
    """Per-feature input |x| range at percentile `pct` and per-hidden-layer activation range at `act_pct`
    (default: `pct`) over the calibration rows X."""
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    in_max = np.percentile(np.abs(x.numpy()), pct, axis=0)
    act_max, h = [], x
    lins = _linears(model)
    for lin in lins[:-1]:
        h = torch.relu(lin.cpu()(h))
        act_max.append(float(np.percentile(h.numpy(), pct if act_pct is None else act_pct)))
    return {"in_max": np.maximum(in_max, 1e-6), "act_max": [max(a, 1e-6) for a in act_max]}


class IntMLP:
    """Frozen integer model built from a float StudentMLP and calibration ranges.
    act_bits = 8 is full int8 (W8A8); act_bits = 16 keeps int8 weights with 16-bit inputs/activations
    (TFLite's "16x8" mode): same weight storage, int16 x int8 products."""

    def __init__(self, model: StudentMLP, cal: dict, act_bits: int = 8):
        lins = [(l.weight.detach().cpu().double().numpy(), l.bias.detach().cpu().double().numpy())
                for l in _linears(model)]
        self.act_bits = act_bits
        self.in_lim, self.act_lim = 2 ** (act_bits - 1) - 1, 2 ** act_bits - 1
        self.s_in = cal["in_max"] / self.in_lim                # per input feature (signed)
        self.s_act = [a / self.act_lim for a in cal["act_max"]]  # per hidden layer (unsigned, after ReLU)
        self.layers = []
        s_prev = None
        for i, (W, b) in enumerate(lins):
            if i == 0:
                Wf = W * self.s_in[None]                       # fold per-feature input scales into W
                s_x = 1.0
            else:
                Wf, s_x = W, s_prev
            s_w = np.maximum(np.abs(Wf).max(1), 1e-12) / 127.0     # per output channel
            Wq = np.clip(np.round(Wf / s_w[:, None]), -127, 127).astype(np.int8)
            acc_scale = s_w * s_x                              # value of one accumulator unit, per channel
            bq = np.round(b / acc_scale).astype(np.int64)      # int32 bias in accumulator units
            self.layers.append((Wq, bq, acc_scale))
            s_prev = self.s_act[i] if i < len(lins) - 1 else None

    def quantize_input(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.round(np.asarray(X, np.float64) / self.s_in), -self.in_lim, self.in_lim).astype(
            np.int8 if self.act_bits == 8 else np.int16)

    def forward(self, X: np.ndarray, batch: int = 262144) -> np.ndarray:
        outs = []
        for i in range(0, len(X), batch):
            h = self.quantize_input(X[i:i + batch]).astype(np.int64)
            for j, (Wq, bq, acc_scale) in enumerate(self.layers):
                acc = h @ Wq.T.astype(np.int64) + bq           # int32 accumulator (int64 here: no overflow)
                if j == len(self.layers) - 1:
                    outs.append((acc * acc_scale).astype(np.float32))
                else:                                          # ReLU + requantise (uint8 / uint16)
                    h = np.clip(np.round(np.maximum(acc * acc_scale, 0) / self.s_act[j]), 0,
                                self.act_lim).astype(np.int64)
        return np.concatenate(outs)

    def size_bytes(self) -> int:
        """int8 weights + int32 biases + float32 scales (per channel, per input feature, per layer)."""
        return int(sum(W.size + 4 * b.size + 4 * s.size for W, b, s in self.layers)
                   + 4 * self.s_in.size + 4 * len(self.s_act))


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _fq(x: torch.Tensor, scale: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(_RoundSTE.apply(x / scale), lo, hi) * scale


class FakeQuantMLP(nn.Module):
    """StudentMLP forward through the IntMLP rounding (straight-through gradients), for QAT."""

    def __init__(self, model: StudentMLP, cal: dict):
        super().__init__()
        self.model = model
        self.register_buffer("s_in", torch.as_tensor(cal["in_max"] / 127.0, dtype=torch.float32))
        self.register_buffer("s_act", torch.as_tensor(np.array(cal["act_max"]) / 255.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _fq(x, self.s_in, -127, 127)
        lins = _linears(self.model)
        for i, lin in enumerate(lins):
            w = lin.weight
            s_w = (w.detach().abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
            h = nn.functional.linear(h, _fq(w, s_w, -127, 127), lin.bias)
            if i < len(lins) - 1:
                h = _fq(torch.relu(h), self.s_act[i], 0, 255)
        return h


def qat_finetune(model: StudentMLP, cal: dict, X: np.ndarray, y: np.ndarray, t_logits: np.ndarray | None,
                 device: torch.device, epochs: int = 3, lr: float = 3e-4, batch: int = 4096, seed: int = 0,
                 max_rows: int = 2_000_000, T: float = 1.0, alpha: float = 0.3, beta: float = 0.7,
                 cols: np.ndarray | None = None) -> StudentMLP:
    """A few epochs of the student's own objective (CE + KL to the teacher when given) through fake-quant,
    on up to `max_rows` training rows (`cols`: the student's input features). Calibration ranges are kept
    fixed (re-measured by the caller afterwards)."""
    g = np.random.default_rng(seed)
    idx = np.sort(g.choice(len(y), min(len(y), max_rows), replace=False))
    Xs = np.asarray(X[idx])
    Xt = torch.as_tensor(Xs if cols is None else Xs[:, cols], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[idx], dtype=torch.long, device=device)
    tt = None if t_logits is None else torch.as_tensor(np.asarray(t_logits[idx]), dtype=torch.float32, device=device)
    fq = FakeQuantMLP(model, cal).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]
            out = fq(Xt[b])
            loss = nn.functional.cross_entropy(out, yt[b])
            if tt is not None:
                loss = alpha * loss + beta * T * T * nn.functional.kl_div(
                    torch.log_softmax(out / T, 1), torch.log_softmax(tt[b] / T, 1), log_target=True,
                    reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()
