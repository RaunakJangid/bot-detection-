"""SHAP explanations of the teacher: global/per-class importance, top-k selection, stability."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from scipy.stats import spearmanr
from torch import nn

from shield.data.datasets import load_processed, stratified_sample
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger(__name__)


class _ClassOutput(nn.Module):
    """View of a classifier that returns only logit `c`, so SHAP computes one output instead of all C."""

    def __init__(self, model: nn.Module, c: int):
        super().__init__()
        self.model, self.c = model, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, self.c:self.c + 1]


def explain(model: nn.Module, background: np.ndarray, X: np.ndarray, y: np.ndarray, device: torch.device,
            nsamples: int, chunk: int) -> np.ndarray:
    """GradientExplainer SHAP values for each row's true class -> (rows, features).

    Rows are grouped by true class and each group is explained through a single-output view of the
    model. Expected gradients treat each output independently, so this is the same estimator as
    explaining all C outputs and keeping the true class, at ~1/C of the cost."""
    model = model.to(device).eval()
    bg = torch.as_tensor(np.ascontiguousarray(background), device=device)
    out = np.zeros(X.shape, dtype=np.float32)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        explainer = shap.GradientExplainer(_ClassOutput(model, int(c)), bg, batch_size=nsamples)
        for i in range(0, len(rows), chunk):
            idx = rows[i:i + chunk]
            xb = torch.as_tensor(np.ascontiguousarray(X[idx]), device=device)
            sv = explainer.shap_values(xb, nsamples=nsamples)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            out[idx] = sv.reshape(len(idx), -1)
    return out


def importance_from_phi(phi: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    glob = np.abs(phi).mean(0)
    per_class = np.zeros((n_classes, phi.shape[1]), np.float32)
    for c in range(n_classes):
        if (y == c).any():
            per_class[c] = np.abs(phi[y == c]).mean(0)
    return glob, per_class


def k_for_cumulative(importance: np.ndarray, threshold: float) -> int:
    s = np.sort(importance)[::-1]
    cum = np.cumsum(s) / max(s.sum(), 1e-12)
    return int(np.searchsorted(cum, threshold) + 1)


def shap_dir(paths: Paths, dataset: str, task: str, model: str, seed: int) -> Path:
    return teacher_dir(paths, dataset, task, model, seed) / "shap"


def run_shap(paths: Paths, dataset: str, task: str, model_name: str, seed: int, cfg: dict,
             device: torch.device, force: bool = False) -> dict:
    out = shap_dir(paths, dataset, task, model_name, seed)
    if (out / "importance.json").exists() and not force:
        log.info("SHAP %s exists, skipping", out)
        return load_json(out / "importance.json")
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_teacher(teacher_dir(paths, dataset, task, model_name, seed), device)
    data = load_processed(paths.processed(dataset), task, splits=("train", "test"))
    tr, te = data.splits["train"], data.splits["test"]
    rng = np.random.default_rng(seed)
    bg_idx = stratified_sample(tr.y, cfg["n_background"], rng, min_per_class=5)
    ex_idx = stratified_sample(te.y, cfg["n_explain"], rng, min_per_class=20)
    X_ex, y_ex = np.asarray(te.X[ex_idx]), te.y[ex_idx]

    phi = explain(model, np.asarray(tr.X[bg_idx]), X_ex, y_ex, device, cfg["nsamples"], cfg["batch_size"])
    glob, per_class = importance_from_phi(phi, y_ex, data.n_classes)
    order = np.argsort(-glob)
    result = {
        "dataset": dataset, "task": task, "seed": seed, "features": data.features,
        "importance": glob, "per_class_importance": per_class, "classes": data.classes,
        "ranking": order, "ranked_features": [data.features[i] for i in order],
        "k95": k_for_cumulative(glob, cfg["cumulative_threshold"]),
        "n_background": len(bg_idx), "n_explain": len(ex_idx), "nsamples": cfg["nsamples"],
    }
    np.save(out / "shap_values.npy", phi)
    np.save(out / "X_explain.npy", X_ex)
    np.save(out / "y_explain.npy", y_ex)
    np.save(out / "explain_idx.npy", ex_idx)
    np.save(out / "background_idx.npy", bg_idx)
    save_json(result, out / "importance.json")
    plot_shap(phi, X_ex, data.features, glob, out, cfg["top_plot"])
    log.info("SHAP %s_%s s%d: k95=%d, top-5 %s", dataset, task, seed, result["k95"],
             result["ranked_features"][:5])
    return result


def plot_shap(phi: np.ndarray, X: np.ndarray, features: list[str], glob: np.ndarray, out: Path, top: int) -> None:
    order = np.argsort(-glob)[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(order) + 1))
    ax.barh([features[i] for i in order], glob[order], color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (true class)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"shap_bar.{ext}", dpi=200)
    plt.close(fig)
    shap.summary_plot(phi, X, feature_names=features, max_display=top, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"shap_beeswarm.{ext}", dpi=200)
    plt.close(fig)


def shap_stability(paths: Paths, dataset: str, task: str, model_name: str, seeds: list[int]) -> dict:
    imps = {}
    for s in seeds:
        f = shap_dir(paths, dataset, task, model_name, s) / "importance.json"
        if f.exists():
            imps[s] = np.asarray(load_json(f)["importance"])
    keys = sorted(imps)
    mat = np.eye(len(keys))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                mat[i, j] = mat[j, i] = spearmanr(imps[a], imps[b]).statistic
    off = mat[np.triu_indices(len(keys), 1)]
    res = {"seeds": keys, "spearman": mat, "mean_spearman": float(off.mean()) if len(off) else None}
    save_json(res, paths.out("shap") / f"{dataset}_{task}_{model_name}_stability.json")
    return res
