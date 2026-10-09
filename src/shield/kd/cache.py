"""One pass of a KD teacher (the teacher, or a teacher assistant) over the data, cached as:
    logits_train.npy   fp16 (n_train, C)  soft targets
    attr_train.npy     fp16 (n_train, F)  gradient x input of the model's PREDICTED class (e2KD)
    pred_test.npy      int16 (n_test,)    test predictions, for student-teacher agreement
Students then train without ever running the teacher, so each student run takes minutes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from shield.data.datasets import load_processed, predict_logits
from shield.kd.losses import grad_x_input
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)

CACHE_VERSION = 2  # v2: predicted-class attributions + test predictions


def cache_dir(paths: Paths, dataset: str, task: str, model: str, seed: int) -> Path:
    return teacher_dir(paths, dataset, task, model, seed) / "cache"


def compute_outputs(model: torch.nn.Module, X: np.ndarray, device: torch.device, logits_out: np.ndarray,
                    attr_out: np.ndarray, batch: int = 16384) -> None:
    """Fill logits and predicted-class gradient x input for every row of X."""
    model.eval()
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch]), device=device)
        with torch.no_grad():
            pred = model(xb).argmax(1)
        attr, logits = grad_x_input(model, xb, pred, create_graph=False)
        logits_out[i:i + batch] = logits.detach().to(torch.float16).cpu().numpy()
        attr_out[i:i + batch] = attr.detach().to(torch.float16).cpu().numpy()


def write_cache(model: torch.nn.Module, X_train: np.ndarray, X_test: np.ndarray, n_classes: int, out: Path,
                device: torch.device, batch: int = 16384) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    n, f = X_train.shape
    t0 = time.time()
    logits = np.lib.format.open_memmap(out / "logits_train.npy", "w+", np.float16, (n, n_classes))
    attr = np.lib.format.open_memmap(out / "attr_train.npy", "w+", np.float16, (n, f))
    compute_outputs(model, X_train, device, logits, attr, batch)
    logits.flush(); attr.flush()
    del logits, attr
    np.save(out / "pred_test.npy", predict_logits(model, X_test, device).argmax(1).astype(np.int16))
    info = {"version": CACHE_VERSION, "attr_class": "predicted", "rows": n, "features": f, "classes": n_classes,
            "seconds": round(time.time() - t0, 1)}
    save_json(info, out / "done.json")
    return info


def cache_ok(out: Path) -> bool:
    f = out / "done.json"
    return f.exists() and load_json(f).get("version") == CACHE_VERSION


def build_cache(paths: Paths, dataset: str, task: str, model_name: str, seed: int, device: torch.device,
                batch: int = 16384, force: bool = False) -> dict:
    out = cache_dir(paths, dataset, task, model_name, seed)
    if cache_ok(out) and not force:
        log.info("Teacher cache %s exists, skipping", out)
        return load_json(out / "done.json")
    model, ckpt = load_teacher(teacher_dir(paths, dataset, task, model_name, seed), device)
    data = load_processed(paths.processed(dataset), task, splits=("train", "test"))
    info = write_cache(model, data.splits["train"].X, data.splits["test"].X, ckpt["n_classes"], out, device, batch)
    log.info("Teacher cache %s_%s s%d: %d rows in %.0fs", dataset, task, seed, info["rows"], info["seconds"])
    return info


def write_logit_cache(logits_train: np.ndarray, pred_test: np.ndarray, out: Path, **info) -> dict:
    """Cache of a teacher that has no input gradients (e.g. an ensemble with trees): soft targets only."""
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "logits_train.npy", logits_train.astype(np.float16))
    np.save(out / "pred_test.npy", pred_test.astype(np.int16))
    info = {"version": CACHE_VERSION, "attr_class": None, "rows": len(logits_train),
            "classes": logits_train.shape[1], **info}
    save_json(info, out / "done.json")
    return info


def load_cache_dir(out: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(train logits, train attributions or None for a logits-only cache, test predictions)."""
    if not cache_ok(out):
        raise FileNotFoundError(f"No v{CACHE_VERSION} cache at {out}; run 05_cache_teacher.py "
                                "(or the teacher-assistant pre-stage) first")
    attr = out / "attr_train.npy"
    return (np.load(out / "logits_train.npy", mmap_mode="r"),
            np.load(attr, mmap_mode="r") if attr.exists() else None, np.load(out / "pred_test.npy"))


def load_cache(paths: Paths, dataset: str, task: str, model_name: str, seed: int):
    return load_cache_dir(cache_dir(paths, dataset, task, model_name, seed))
