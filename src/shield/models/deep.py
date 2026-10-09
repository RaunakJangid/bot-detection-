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


def build_deep(name: str, n_features: int, n_classes: int) -> nn.Module:
    if name in ("cnn_lstm", "cnn_gru"):
        return ConvRecurrent(n_features, n_classes, cell=name.split("_")[1])
    if name == "ftt":
        from shield.models.teacher import FTTransformer
        return FTTransformer(n_features, n_classes, d_token=64, n_layers=3, n_heads=8, dropout=0.1)
    raise ValueError(name)
