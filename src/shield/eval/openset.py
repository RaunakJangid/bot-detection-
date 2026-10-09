"""Novelty scores for attacks never seen in training (v3 step 8).

Both are fitted on BENIGN training traffic only, so they need no attack labels:
- Mahalanobis (Lee et al., NeurIPS 2018): distance of the student's last hidden layer to the benign
  class mean, with a shrunk covariance.
- FlyHash novelty filter (Dasgupta et al., Science 2017; PNAS 2018): the fruit-fly olfactory circuit.
  A sparse binary random projection expands the d inputs to m >> d "Kenyon cells"; winner-take-all keeps
  the top `wta` cells as the input's tag. A "Bloom filter" over the m cells remembers which cells benign
  inputs have activated; an input is novel when its tag lands on cells benign traffic rarely activated.
  The detector is an m-vector of counters (int16 here), so it fits next to the tiny student.
`union_score` merges the student's attack probability with a novelty score: each is mapped to its
benign-validation exceedance level and the larger one is used, so one threshold sets the joint FPR.
"""

from __future__ import annotations

import numpy as np
import torch


class Mahalanobis:
    def __init__(self, shrink: float = 0.01):
        self.shrink = shrink

    def fit(self, H: np.ndarray) -> "Mahalanobis":
        H = H.astype(np.float64)
        self.mean = H.mean(0)
        cov = np.cov(H, rowvar=False)
        cov += self.shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        d = H.astype(np.float64) - self.mean
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


class FlyHash:
    """Fruit-fly novelty filter. m Kenyon cells, each sampling `fan_in` random inputs (binary projection);
    winner-take-all keeps the top `wta` cells; novelty = mean over the tag of 1 / (1 + log1p(count))."""

    def __init__(self, d: int, m: int = 2000, fan_in: int = 6, wta: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.idx = np.stack([rng.choice(d, min(fan_in, d), replace=False) for _ in range(m)])   # (m, fan_in)
        self.P = np.zeros((d, m), np.float32)                    # the same projection as a 0/1 matrix
        self.P[self.idx, np.arange(m)[:, None]] = 1.0
        self.m, self.wta = m, wta
        self.counts = np.zeros(m, np.int64)

    def _tags(self, X: np.ndarray, batch: int = 50_000) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            x = np.asarray(X[i:i + batch], np.float32)
            x = x - x.mean(1, keepdims=True)                     # mean-centring, as in the fly's PN layer
            act = x @ self.P                                     # (b, m) sparse binary projection
            out.append(np.argpartition(-act, self.wta, axis=1)[:, :self.wta])
        return np.concatenate(out)

    def fit(self, X: np.ndarray) -> "FlyHash":
        tags = self._tags(X)
        self.counts = np.bincount(tags.ravel(), minlength=self.m)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        tags = self._tags(X)
        return (1.0 / (1.0 + np.log1p(self.counts[tags]))).mean(1)

    def size_bytes(self) -> int:
        return int(self.idx.size * 2 + self.m * 2)               # uint16 input indices + int16 counters


def exceedance(score_ref: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Fraction of reference (benign validation) scores at or above each score: low = unusual."""
    ref = np.sort(score_ref)
    return 1.0 - np.searchsorted(ref, score, side="left") / len(ref)


def union_score(p_attack_ref, nov_ref, p_attack, nov) -> np.ndarray:
    """1 - min(exceedance): flagged when EITHER the classifier or the novelty score is extreme vs benign."""
    return 1.0 - np.minimum(exceedance(p_attack_ref, p_attack), exceedance(nov_ref, nov))


def threshold_at_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(benign_scores, 1.0 - fpr))


@torch.no_grad()
def hidden_features(model, X: np.ndarray, device: torch.device, batch: int = 65536) -> np.ndarray:
    """Activations entering the student's output layer."""
    body = model.net[:-1].to(device).eval()
    outs = []
    for i in range(0, len(X), batch):
        outs.append(body(torch.as_tensor(np.ascontiguousarray(X[i:i + batch]), device=device)).float().cpu())
    return torch.cat(outs).numpy()
