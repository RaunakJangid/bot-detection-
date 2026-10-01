"""Shared training loop for teachers and students."""

from __future__ import annotations

import copy
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from shield.data.datasets import Batcher, predict_logits
from shield.eval.metrics import classification_metrics
from shield.utils.logging import get_logger

log = get_logger(__name__)

LossFn = Callable[[nn.Module, list[torch.Tensor]], torch.Tensor]


def warmup_cosine(total_steps: int, warmup_frac: float) -> Callable[[int], float]:
    warmup = max(1, int(total_steps * warmup_frac))

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def fit(model: nn.Module, batcher: Batcher, val_X: np.ndarray, val_y: np.ndarray, loss_fn: LossFn,
        cfg: dict, device: torch.device, n_classes: int, name: str = "model") -> tuple[dict, list[dict]]:
    """Train with AdamW + warmup/cosine; early-stop on validation macro-F1. Returns (best_state, history)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(batcher)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, cfg.get("warmup_frac", 0.03)))
    amp = bool(cfg.get("amp")) and device.type == "cuda"
    best_f1, best_state, bad, history = -1.0, None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, steps = time.time(), 0.0, 0
        for batch in batcher:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() if steps % 50 == 0 else 0.0
            steps += 1
        logits = predict_logits(model, val_X, device, cfg.get("eval_batch_size", 65536), amp=amp)
        m = classification_metrics(val_y, logits, n_classes)
        rec = {"epoch": epoch, "seconds": round(time.time() - t0, 2), "val_macro_f1": m["macro_f1"],
               "val_accuracy": m["accuracy"], "loss_sampled": loss_sum / max(1, math.ceil(steps / 50))}
        history.append(rec)
        log.info("%s epoch %d | %.1fs | val macro-F1 %.4f acc %.4f", name, epoch, rec["seconds"],
                 m["macro_f1"], m["accuracy"])
        if m["macro_f1"] > best_f1 + 1e-5:
            best_f1, bad = m["macro_f1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info("%s early stop at epoch %d (best val macro-F1 %.4f)", name, epoch, best_f1)
                break
    model.load_state_dict(best_state)
    return best_state, history
