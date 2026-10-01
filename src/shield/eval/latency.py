"""Inference cost: single-core CPU latency/throughput (edge proxy), GPU throughput, size, FLOPs."""

from __future__ import annotations

import io
import time

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def model_size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


def flops_per_sample(model: nn.Module, n_features: int) -> int:
    """Forward FLOPs for one sample (float model)."""
    model = model.cpu().eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, n_features))
    return int(fc.get_total_flops())


class _PinnedCore:
    """Run on one logical CPU with one intra-op thread; restores both afterwards."""

    def __init__(self, core: int = 0):
        self.core = core

    def __enter__(self):
        self.proc = psutil.Process()
        try:
            self.prev_aff = self.proc.cpu_affinity()
            self.proc.cpu_affinity([self.core])
        except (AttributeError, psutil.Error):
            self.prev_aff = None
        self.prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        return self

    def __exit__(self, *exc):
        torch.set_num_threads(self.prev_threads)
        if self.prev_aff is not None:
            self.proc.cpu_affinity(self.prev_aff)


@torch.inference_mode()
def cpu_latency(model: nn.Module, n_features: int, batch_sizes=(1, 256), runs: int = 1000,
                warmup: int = 100, core: int = 0) -> dict:
    model = model.cpu().eval()
    out = {}
    with _PinnedCore(core):
        for b in batch_sizes:
            x = torch.randn(b, n_features)
            for _ in range(warmup):
                model(x)
            times = np.empty(runs)
            for i in range(runs):
                t0 = time.perf_counter()
                model(x)
                times[i] = time.perf_counter() - t0
            med = float(np.median(times))
            out[f"latency_b{b}_ms"] = med * 1e3
            out[f"p95_b{b}_ms"] = float(np.percentile(times, 95)) * 1e3
            out[f"throughput_b{b}"] = b / med
    return out


@torch.inference_mode()
def gpu_throughput(model: nn.Module, n_features: int, batch: int = 65536, runs: int = 50) -> float | None:
    if not torch.cuda.is_available():
        return None
    model = model.cuda().eval()
    x = torch.randn(batch, n_features, device="cuda")
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    torch.cuda.synchronize()
    return batch * runs / (time.perf_counter() - t0)
