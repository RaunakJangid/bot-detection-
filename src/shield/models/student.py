"""The lightweight student and its int8 export."""

from __future__ import annotations

import copy
import warnings

import torch
from torch import nn


class StudentMLP(nn.Module):
    """k -> hidden... -> C with ReLU (quantisation friendly, no normalisation layers)."""

    def __init__(self, n_features: int, n_classes: int, hidden: list[int] = (64, 32)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def quantize_int8(model: nn.Module) -> nn.Module:
    """Post-training dynamic int8 quantisation of all Linear layers (CPU inference)."""
    model = copy.deepcopy(model).cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization emits deprecation notices on newer torch
        from torch.ao.quantization import quantize_dynamic
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
