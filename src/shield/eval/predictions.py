"""Validation/test logits of finished v2 models, cached under the v3 root (float16) for the v3 steps.

A model is named "teacher", "xgboost", a v2 KD run name (e.g. "kd_all", "shield"), or "v3-<run>" for a
student trained by the v3 scripts (outputs_v3/kd).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shield.data.datasets import ProcessedData, predict_logits
from shield.kd.trainer import kd_dir
from shield.models.student import StudentMLP
from shield.models.train import load_teacher, teacher_dir
from shield.models.xgb import xgb_logits
from shield.utils.config import Paths


def run_dir(paths: Paths, dataset: str, task: str, model: str, seed: int, teacher_model: str = "resmlp") -> Path:
    if model == "teacher":
        return teacher_dir(paths, dataset, task, teacher_model, seed)
    if model == "xgboost":
        return paths.outputs / "xgboost" / f"{dataset}_{task}_s{seed}"
    if model.startswith("v3-"):
        return kd_dir(paths, dataset, task, model[3:], seed, base=paths.v3)
    return kd_dir(paths, dataset, task, model, seed)


def load_student(run: Path, device: torch.device) -> tuple[StudentMLP, np.ndarray]:
    ckpt = torch.load(run / "student.pt", map_location="cpu", weights_only=False)
    S = np.asarray(ckpt["selected"])
    model = StudentMLP(len(S), ckpt["n_classes"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), S


def model_logits(paths: Paths, data: ProcessedData, model: str, seed: int, splits: tuple[str, ...],
                 device: torch.device, teacher_model: str = "resmlp") -> dict[str, np.ndarray] | None:
    """{split: logits (float32)}; None if the model's run does not exist."""
    run = run_dir(paths, data.dataset, data.task, model, seed, teacher_model)
    cache = paths.v3 / "logits" / f"{data.dataset}_{data.task}"
    files = {s: cache / f"{model}_s{seed}_{s}.npy" for s in splits}
    if all(f.exists() for f in files.values()):
        return {s: np.load(f).astype(np.float32) for s, f in files.items()}
    marker = "model.pt" if model == "teacher" else "model.json" if model == "xgboost" else "student.pt"
    if not (run / marker).exists():
        return None
    if model == "teacher":
        net, _ = load_teacher(run, device)
        fn = lambda X: predict_logits(net, X, device, 65536, amp=True)
    elif model == "xgboost":
        fn = lambda X: xgb_logits(run, X, data.n_classes)
    else:
        net, S = load_student(run, device)
        fn = lambda X: predict_logits(net, X if len(S) == X.shape[1] else np.asarray(X[:, S]), device, 65536)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for s, f in files.items():
        if f.exists():
            out[s] = np.load(f).astype(np.float32)
            continue
        out[s] = fn(data.splits[s].X).astype(np.float32)
        np.save(f, out[s].astype(np.float16))
    return out
