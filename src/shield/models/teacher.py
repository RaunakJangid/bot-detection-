"""Teacher networks for tabular flow features."""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResMLP(nn.Module):
    """Residual MLP: input projection -> `depth` pre-norm residual blocks -> head."""

    def __init__(self, n_features: int, n_classes: int, width: int = 384, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.BatchNorm1d(width), nn.GELU(), nn.Linear(width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.inp(x)))


class FTTransformer(nn.Module):
    """Compact FT-Transformer: per-feature linear tokens + [CLS] + Transformer encoder."""

    def __init__(self, n_features: int, n_classes: int, d_token: int = 64, n_layers: int = 3,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * d_token ** -0.5)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * d_token ** -0.5)
        layer = nn.TransformerEncoderLayer(d_token, n_heads, 2 * d_token, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0])


def build_teacher(name: str, n_features: int, n_classes: int, cfg: dict) -> nn.Module:
    if name == "resmlp":
        return ResMLP(n_features, n_classes, **cfg["resmlp"])
    if name == "ftt":
        return FTTransformer(n_features, n_classes, **cfg["ftt"])
    raise ValueError(f"Unknown teacher model {name!r}")
