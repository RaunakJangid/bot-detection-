"""One pass of a teacher over the training split: fp16 logits + true-class gradient x input.

Students then train without ever running the teacher, so each student run takes minutes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from shield.data.datasets import load_processed
from shield.kd.losses import grad_x_input
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)


def cache_dir(paths: Paths, dataset: str, task: str, model: str, seed: int) -> Path:
    return teacher_dir(paths, dataset, task, model, seed) / "cache"


def compute_teacher_outputs(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device,
                            logits_out: np.ndarray, attr_out: np.ndarray, batch: int = 16384) -> None:
    model.eval()
    for i in range(0, len(y), batch):
        xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch]), device=device)
        yb = torch.as_tensor(y[i:i + batch], device=device)
        attr, logits = grad_x_input(model, xb, yb, create_graph=False)
        logits_out[i:i + batch] = logits.detach().to(torch.float16).cpu().numpy()
        attr_out[i:i + batch] = attr.detach().to(torch.float16).cpu().numpy()


def build_cache(paths: Paths, dataset: str, task: str, model_name: str, seed: int, device: torch.device,
                batch: int = 16384, force: bool = False) -> dict:
    out = cache_dir(paths, dataset, task, model_name, seed)
    if (out / "done.json").exists() and not force:
        log.info("Teacher cache %s exists, skipping", out)
        return load_json(out / "done.json")
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_teacher(teacher_dir(paths, dataset, task, model_name, seed), device)
    tr = load_processed(paths.processed(dataset), task, splits=("train",)).splits["train"]
    n, f = tr.X.shape
    t0 = time.time()
    logits = np.lib.format.open_memmap(out / "logits_train.npy", "w+", np.float16, (n, ckpt["n_classes"]))
    attr = np.lib.format.open_memmap(out / "attr_train.npy", "w+", np.float16, (n, f))
    compute_teacher_outputs(model, tr.X, tr.y, device, logits, attr, batch)
    logits.flush(); attr.flush()
    del logits, attr
    info = {"rows": n, "features": f, "classes": ckpt["n_classes"], "seconds": round(time.time() - t0, 1)}
    save_json(info, out / "done.json")
    log.info("Teacher cache %s_%s s%d: %d rows in %.0fs", dataset, task, seed, n, info["seconds"])
    return info


def load_cache(paths: Paths, dataset: str, task: str, model_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    out = cache_dir(paths, dataset, task, model_name, seed)
    if not (out / "done.json").exists():
        raise FileNotFoundError(f"No teacher cache at {out}; run 05_cache_teacher.py first")
    return np.load(out / "logits_train.npy", mmap_mode="r"), np.load(out / "attr_train.npy", mmap_mode="r")
