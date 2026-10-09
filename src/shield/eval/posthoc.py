"""Post-hoc decision rules for macro-F1, fitted on validation only (v3 step 2).

Two rules on top of a model's log-probabilities z (no retraining):
- logit adjustment (Menon et al., ICLR 2021): z - tau * log(prior), prior = training class frequencies,
  tau chosen on validation from a grid (tau = 0 is the unadjusted model, so it can only be kept or improved);
- per-class bias b on top of that, by coordinate ascent on validation macro-F1.
Both are a fixed vector added to the logits, so they cost nothing at inference time.
"""

from __future__ import annotations

import numpy as np
import torch

TAUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BIAS_GRID = (-2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _macro_f1(y: torch.Tensor, pred: torch.Tensor, n: int, present: torch.Tensor) -> float:
    conf = torch.bincount(y * n + pred, minlength=n * n).reshape(n, n).double()
    tp = conf.diagonal()
    f1 = 2 * tp / (conf.sum(0) + conf.sum(1)).clamp(min=1)
    return float(f1[present].mean())   # classes absent from these labels are left out, as in sklearn


class _Scorer:
    def __init__(self, logp: np.ndarray, y: np.ndarray, device: torch.device):
        self.z = torch.as_tensor(logp, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.n = logp.shape[1]
        self.present = torch.bincount(self.y, minlength=self.n) > 0

    def __call__(self, bias: torch.Tensor) -> float:
        return _macro_f1(self.y, (self.z + bias).argmax(1), self.n, self.present)


def fit_decision_rule(val_logits: np.ndarray, y_val: np.ndarray, prior: np.ndarray,
                      device: torch.device | None = None, sweeps: int = 3) -> dict:
    """Returns {"tau", "la_bias", "bias", "val_*"}; `la_bias` is logit adjustment alone, `bias` adds the
    per-class terms. Apply with `apply_bias`."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = _Scorer(log_softmax(val_logits.astype(np.float32)), y_val, device)
    log_prior = torch.as_tensor(np.log(np.clip(prior, 1e-12, None)), dtype=torch.float32, device=device)

    raw = score(torch.zeros(score.n, device=device))
    tau, best = 0.0, raw
    for t in TAUS[1:]:
        f = score(-t * log_prior)
        if f > best + 1e-6:
            tau, best = t, f
    la = -tau * log_prior
    la_f1 = best

    b = la.clone()
    for _ in range(sweeps):
        improved = False
        for c in range(score.n):
            if not score.present[c]:
                continue
            base = b[c].item()
            for d in BIAS_GRID:
                b[c] = base + d
                f = score(b)
                if f > best + 1e-6:
                    best, base, improved = f, base + d, True
            b[c] = base
        if not improved:
            break
    return {"tau": tau, "la_bias": la.cpu().numpy(), "bias": b.cpu().numpy(),
            "val_macro_f1_raw": raw, "val_macro_f1_la": la_f1, "val_macro_f1_bias": best}


def apply_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return log_softmax(logits.astype(np.float32)) + bias[None].astype(np.float32)
