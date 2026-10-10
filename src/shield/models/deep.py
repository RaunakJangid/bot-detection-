"""Deep baselines from the IoT-IDS literature, re-implemented for strict-protocol comparison (v3):
the flow-feature vector is treated as a 1-D sequence (one step per feature), as in CNN-LSTM / CNN-GRU IDS papers
(e.g. AttackNet, Nandanwar et al. 2024). FT-Transformer is shield.models.teacher.FTTransformer."""

from __future__ import annotations

import torch
from torch import nn


class ConvRecurrent(nn.Module):
    """Conv1d(1->c) x2 + max-pool over the feature axis, then a recurrent layer (LSTM or GRU), then a head."""

    def __init__(self, n_features: int, n_classes: int, cell: str = "lstm", channels: int = 64, hidden: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(),
                                  nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(),
                                  nn.MaxPool1d(2))
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)      # (batch, steps, channels)
        out, _ = self.rnn(h)
        return self.head(out[:, -1])


class TabM(nn.Module):
    """TabM (Gorishniy et al., ICLR 2025): an MLP whose k members share every weight matrix W and differ only by
    rank-1 multiplicative adapters, h_i = s_i * (W (r_i * x_i)) + b_i (BatchEnsemble). Training: mean of the members'
    losses (forward returns (batch, k, classes)); inference: log of the members' mean probability."""

    def __init__(self, n_features: int, n_classes: int, k: int = 32, hidden: int = 512, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.k = k
        dims = [n_features] + [hidden] * n_layers
        self.W = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        init = lambda n: nn.Parameter(1 + 0.5 * (2 * torch.rand(k, n) - 1))   # random-sign init around 1
        self.r = nn.ParameterList(init(a) for a in dims[:-1])
        self.s = nn.ParameterList(init(b) for b in dims[1:])
        self.b = nn.ParameterList(nn.Parameter(torch.zeros(k, b)) for b in dims[1:])
        self.drop = nn.Dropout(dropout)
        self.head_w = nn.Parameter(torch.randn(k, hidden, n_classes) / hidden ** 0.5)
        self.head_b = nn.Parameter(torch.zeros(k, n_classes))

    def members(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1).expand(-1, self.k, -1)
        for W, r, s, b in zip(self.W, self.r, self.s, self.b):
            h = self.drop(torch.relu(W(h * r) * s + b))
        return torch.einsum("bkh,khc->bkc", h, self.head_w) + self.head_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.members(x)
        if self.training:
            return z
        return torch.logsumexp(torch.log_softmax(z.float(), -1), 1) - torch.log(torch.tensor(float(self.k)))


class TabMStudent(TabM):
    """Edge-sized TabM: k members share every weight matrix AND the head (members differ only by the rank-1
    adapters r, s, b and a head bias), so storage stays near a plain MLP of the same width (k x the FLOPs)."""

    def __init__(self, n_features: int, n_classes: int, k: int = 8, hidden=(128, 64)):
        nn.Module.__init__(self)
        self.k = k
        dims = [n_features, *hidden]
        self.W = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        init = lambda n: nn.Parameter(1 + 0.5 * (2 * torch.rand(k, n) - 1))
        self.r = nn.ParameterList(init(a) for a in dims[:-1])
        self.s = nn.ParameterList(init(b) for b in dims[1:])
        self.b = nn.ParameterList(nn.Parameter(torch.zeros(k, b)) for b in dims[1:])
        self.drop = nn.Identity()
        self.head = nn.Linear(dims[-1], n_classes)
        self.head_b = nn.Parameter(torch.zeros(k, n_classes))

    def members(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1).expand(-1, self.k, -1)
        for W, r, s, b in zip(self.W, self.r, self.s, self.b):
            h = torch.relu(W(h * r) * s + b)
        return self.head(h) + self.head_b


def build_deep(name: str, n_features: int, n_classes: int) -> nn.Module:
    if name == "tabm":
        return TabM(n_features, n_classes)
    if name == "tabm_s":     # ~21k parameters
        return TabMStudent(n_features, n_classes)
    if name == "mlp_s":      # parameter-matched plain MLP student (~20k), same recipe
        from shield.models.student import StudentMLP
        return StudentMLP(n_features, n_classes, hidden=[144, 72])
    if name in ("cnn_lstm", "cnn_gru"):
        return ConvRecurrent(n_features, n_classes, cell=name.split("_")[1])
    if name == "ftt":
        from shield.models.teacher import FTTransformer
        return FTTransformer(n_features, n_classes, d_token=64, n_layers=3, n_heads=8, dropout=0.1)
    raise ValueError(name)
