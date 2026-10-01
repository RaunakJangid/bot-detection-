"""Teacher training runs (one per dataset / task / seed) and teacher loading."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shield.data.datasets import Batcher, class_weights, load_processed, predict_logits
from shield.data.registry import Registry
from shield.engine import fit
from shield.eval.metrics import classification_metrics, confusion
from shield.models.student import count_params
from shield.models.teacher import build_teacher
from shield.utils.config import Paths, deep_merge
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything

log = get_logger(__name__)


def teacher_dir(paths: Paths, dataset: str, task: str, model: str, seed: int) -> Path:
    return paths.outputs / "teacher" / f"{dataset}_{task}_{model}_s{seed}"


def train_cfg_for(cfg: dict, dataset: str) -> dict:
    """`train` merged with the overrides of the dataset's raw source (e.g. botiot_std uses botiot's)."""
    source = Registry().source(dataset)
    return deep_merge(cfg["train"], (cfg.get("dataset_overrides") or {}).get(source, {}))


def train_teacher(paths: Paths, dataset: str, task: str, seed: int, cfg: dict, device: torch.device,
                  force: bool = False) -> dict:
    out = teacher_dir(paths, dataset, task, cfg["model"], seed)
    if (out / "metrics.json").exists() and not force:
        log.info("Teacher %s exists, skipping", out.name)
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    tcfg = train_cfg_for(cfg, dataset)
    data = load_processed(paths.processed(dataset), task)
    tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]

    model = build_teacher(cfg["model"], data.n_features, data.n_classes, cfg)
    cw = class_weights(tr.y, data.n_classes, tcfg["class_weight"])
    ce = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw, device=device))
    batcher = Batcher(tr.X, tr.y, tcfg["batch_size"], device, balance_power=tcfg["balance_power"],
                      samples_per_epoch=tcfg["samples_per_epoch"], seed=seed)
    log.info("Teacher %s: %d params, train rows %d, data on device: %s", out.name, count_params(model),
             len(tr.y), batcher.on_device)

    t0 = time.time()
    state, history = fit(model, batcher, va.X, va.y, lambda m, b: ce(m(b[0]), b[1]), tcfg, device,
                         data.n_classes, name=out.name)
    train_seconds = time.time() - t0
    del batcher

    torch.save({"state_dict": state, "model": cfg["model"], "n_features": data.n_features,
                "n_classes": data.n_classes, "arch": cfg[cfg["model"]], "features": data.features,
                "classes": data.classes}, out / "model.pt")
    logits = predict_logits(model, te.X, device, tcfg["eval_batch_size"], amp=bool(tcfg.get("amp")))
    np.save(out / "confusion.npy", confusion(te.y, logits, data.n_classes))
    metrics = {
        "dataset": dataset, "task": task, "seed": seed, "model": cfg["model"],
        "params": count_params(model), "train_seconds": round(train_seconds, 1),
        "test": classification_metrics(te.y, logits, data.n_classes, data.classes, full=True),
        "history": history, "train_cfg": tcfg, "hardware": hardware_record(),
    }
    save_json(metrics, out / "metrics.json")
    log.info("Teacher %s test macro-F1 %.4f acc %.4f", out.name, metrics["test"]["macro_f1"],
             metrics["test"]["accuracy"])
    return metrics


def load_teacher(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = {ckpt["model"]: ckpt["arch"]}
    model = build_teacher(ckpt["model"], ckpt["n_features"], ckpt["n_classes"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
