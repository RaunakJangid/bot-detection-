"""Loading processed arrays and feeding them to PyTorch.

Small datasets (CICIoT2023, ~1.5 GB) live entirely on the GPU and are indexed there.
Large ones (Bot-IoT) stay memory-mapped on the host; batches are gathered with sorted
fancy indexing, pinned, and copied asynchronously by a prefetch thread.
"""

from __future__ import annotations

import queue
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from shield.utils.io import load_json

SPLITS = ("train", "validation", "test")


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray


@dataclass
class ProcessedData:
    dataset: str
    task: str
    classes: list[str]
    features: list[str]
    meta: dict
    splits: dict[str, Split] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_features(self) -> int:
        return len(self.features)


def load_processed(ds_dir: Path, task: str, mmap: bool = True, splits=SPLITS) -> ProcessedData:
    meta = load_json(ds_dir / "meta.json")
    if task not in meta["tasks"]:
        raise ValueError(f"Task {task!r} not in {list(meta['tasks'])}")
    data = ProcessedData(meta["dataset"], task, meta["tasks"][task], meta["features"], meta)
    for s in splits:
        X = np.load(ds_dir / f"X_{s}.npy", mmap_mode="r" if mmap else None)
        y = np.load(ds_dir / f"y_{task}_{s}.npy").astype(np.int64)
        data.splits[s] = Split(X, y)
    return data


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    if mode in (None, "none"):
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts) if mode == "inverse_sqrt" else 1.0 / counts
    return (w / (w * counts).sum() * counts.sum()).astype(np.float32)  # mean weight per sample = 1


def stratified_sample(y: np.ndarray, n: int, rng: np.random.Generator, min_per_class: int = 1) -> np.ndarray:
    """~n indices, proportional to class frequency but with at least `min_per_class` of every class."""
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(np.round(counts / counts.sum() * n).astype(int), min_per_class)
    quota = np.minimum(quota, counts)
    picks = [rng.choice(np.flatnonzero(y == c), q, replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(picks))


def gpu_fits(arrays: list[np.ndarray], device: torch.device, frac: float = 0.6) -> bool:
    if device.type != "cuda":
        return True  # host memory: tensors share the numpy buffers
    need = sum(a.nbytes for a in arrays)
    free, _ = torch.cuda.mem_get_info(device)
    return need < frac * free


class Batcher:
    """Iterates (x, y, *extras) mini-batches on `device`.

    balance_power p re-weights sampling by count_c ** -p (0 = uniform shuffle, 1 = class-balanced);
    samples_per_epoch fixes the epoch length (with replacement when p > 0).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device,
                 shuffle: bool = True, balance_power: float = 0.0, samples_per_epoch: int | None = None,
                 extras: list[np.ndarray] | None = None, seed: int = 0, on_device: bool | None = None,
                 prefetch: int = 4):
        self.batch_size, self.device, self.shuffle = batch_size, device, shuffle
        self.balance_power, self.samples_per_epoch = balance_power, samples_per_epoch
        self.n = len(y)
        self.rng = np.random.default_rng(seed)
        self.y_np = np.asarray(y)
        arrays = [X, y] + list(extras or [])
        self.on_device = gpu_fits(arrays, device) if on_device is None else on_device
        self.prefetch = prefetch
        if self.on_device:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # read-only memmaps; tensors are never written
                self.tensors = [torch.as_tensor(np.ascontiguousarray(a)).to(device) for a in arrays]
        else:
            self.arrays = arrays
        if balance_power > 0:
            counts = np.bincount(self.y_np).astype(np.float64)
            w = np.zeros_like(counts)
            w[counts > 0] = counts[counts > 0] ** -balance_power
            p = w[self.y_np]
            self.p = p / p.sum()
        else:
            self.p = None

    def _epoch_indices(self) -> np.ndarray:
        m = self.samples_per_epoch or self.n
        if self.p is not None:
            return self.rng.choice(self.n, size=m, replace=True, p=self.p)
        if not self.shuffle:
            return np.arange(self.n)
        if m <= self.n:
            return self.rng.permutation(self.n)[:m]
        return self.rng.integers(0, self.n, size=m)

    def __len__(self) -> int:
        return -(-(self.samples_per_epoch or self.n) // self.batch_size)

    def _host_batch(self, idx: np.ndarray) -> list[torch.Tensor]:
        idx = np.sort(idx)  # sorted gathers are much faster on memory-mapped arrays
        out = []
        for a in self.arrays:
            t = torch.from_numpy(np.ascontiguousarray(a[idx]))
            if self.device.type == "cuda":
                t = t.pin_memory().to(self.device, non_blocking=True)
            out.append(t)
        return out

    def __iter__(self):
        order = self._epoch_indices()
        starts = range(0, len(order), self.batch_size)
        if self.on_device:
            idx_all = torch.as_tensor(order, device=self.device)
            for s in starts:
                idx = idx_all[s:s + self.batch_size]
                yield [t[idx] for t in self.tensors]
            return
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def worker():
            try:
                for s in starts:
                    if stop.is_set():
                        return
                    q.put(self._host_batch(order[s:s + self.batch_size]))
            except Exception as exc:  # surface loader errors in the training thread
                q.put(exc)
            q.put(None)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            while not q.empty():
                q.get_nowait()


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 65536,
                   amp: bool = False) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            outs.append(model(xb).float().cpu())
    return torch.cat(outs).numpy()
