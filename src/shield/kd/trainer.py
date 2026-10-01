"""One distillation run = (dataset, task, ablation variant, seed)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from shield.data.datasets import Batcher, class_weights, load_processed, predict_logits, stratified_sample
from shield.engine import fit
from shield.eval.metrics import classification_metrics, confusion
from shield.kd.cache import load_cache
from shield.kd.losses import ShieldLoss
from shield.models.student import StudentMLP, count_params, quantize_int8
from shield.models.train import train_cfg_for
from shield.utils.config import Paths, deep_merge
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything
from shield.xai.selection import select_features
from shield.xai.shap_analysis import explain, importance_from_phi, shap_dir

log = get_logger(__name__)


def kd_dir(paths: Paths, dataset: str, task: str, variant: str, seed: int) -> Path:
    return paths.outputs / "kd" / f"{dataset}_{task}" / f"{variant}_s{seed}"


def _columns(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    return X if len(S) == X.shape[1] else np.asarray(X[:, S])


def run_distillation(paths: Paths, dataset: str, task: str, variant: str, seed: int, cfg: dict,
                     teacher_model: str, device: torch.device, force: bool = False,
                     k_override: int | None = None) -> dict:
    """k_override (for the F1-vs-k sweep) stores the run as '<variant>_k<k>'."""
    run_name = variant if k_override is None else f"{variant}_k{k_override}"
    out = kd_dir(paths, dataset, task, run_name, seed)
    if (out / "metrics.json").exists() and not force:
        log.info("KD %s/%s exists, skipping", out.parent.name, out.name)
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    v = cfg["variants"][variant]
    loss_cfg = deep_merge(cfg["loss"], {k: v[k] for k in ("alpha", "beta", "gamma", "T") if k in v})
    tcfg = train_cfg_for(cfg, dataset)

    data = load_processed(paths.processed(dataset), task)
    tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
    t_seed = cfg["teacher_seed"]
    shap_info = load_json(shap_dir(paths, dataset, task, teacher_model, t_seed) / "importance.json")
    importance = np.asarray(shap_info["importance"], dtype=np.float32)
    k_cfg = cfg["k"] if k_override is None else k_override
    k = shap_info["k95"] if k_cfg == "k95" else int(k_cfg)

    rng = np.random.default_rng(seed)
    sample_idx = stratified_sample(tr.y, min(cfg["mi_sample"], len(tr.y)), rng, min_per_class=10)
    S = select_features(v["select"], k, data.n_features, shap_ranking=np.asarray(shap_info["ranking"]),
                        X_sample=np.asarray(tr.X[sample_idx]), y_sample=tr.y[sample_idx],
                        meta=data.meta, seed=seed)
    w = importance[S] / max(importance[S].max(), 1e-12)

    loss = ShieldLoss(loss_cfg["alpha"], loss_cfg["beta"], loss_cfg["gamma"], loss_cfg["T"],
                      feature_weights=torch.as_tensor(w, device=device))
    cw = class_weights(tr.y, data.n_classes, tcfg["class_weight"])
    if cw is not None:
        loss.ce.weight = torch.as_tensor(cw, device=device)
    extras = []
    if loss.needs_logits or loss.needs_attr:
        t_logits, t_attr = load_cache(paths, dataset, task, teacher_model, t_seed)
        if loss.needs_logits:
            extras.append(t_logits)
        if loss.needs_attr:
            extras.append(_columns(t_attr, S))

    model = StudentMLP(len(S), data.n_classes, cfg["student"]["hidden"])
    batcher = Batcher(_columns(tr.X, S), tr.y, tcfg["batch_size"], device, balance_power=tcfg["balance_power"],
                      samples_per_epoch=tcfg["samples_per_epoch"], extras=extras, seed=seed)
    name = f"{dataset}_{task}/{run_name}_s{seed}"
    t0 = time.time()
    state, history = fit(model, batcher, _columns(va.X, S), va.y, loss, tcfg, device, data.n_classes, name)
    train_seconds = time.time() - t0
    del batcher, extras

    X_te = _columns(te.X, S)
    logits = predict_logits(model, X_te, device, tcfg["eval_batch_size"])
    np.save(out / "confusion.npy", confusion(te.y, logits, data.n_classes))
    q_model = quantize_int8(model)
    q_logits = predict_logits(q_model, X_te, torch.device("cpu"), tcfg["eval_batch_size"])

    torch.save({"state_dict": state, "selected": S, "features": [data.features[i] for i in S],
                "hidden": cfg["student"]["hidden"], "n_classes": data.n_classes, "classes": data.classes},
               out / "student.pt")
    torch.save(q_model, out / "student_int8.pt")

    metrics = {
        "dataset": dataset, "task": task, "variant": variant, "run": run_name, "seed": seed, "k": int(len(S)),
        "selection": v["select"], "loss": loss_cfg, "selected": S,
        "selected_features": [data.features[i] for i in S], "params": count_params(model),
        "train_seconds": round(train_seconds, 1),
        "test": classification_metrics(te.y, logits, data.n_classes, data.classes, full=True),
        "test_int8": classification_metrics(te.y, q_logits, data.n_classes, data.classes, full=True),
        "history": history, "hardware": hardware_record(),
    }
    fid = cfg["fidelity"]
    metrics["fidelity"] = (student_fidelity(paths, dataset, task, teacher_model, t_seed, model, S, tr.X,
                                            importance, fid, device, seed)
                           if seed in fid["seeds"] else None)
    save_json(metrics, out / "metrics.json")
    log.info("KD %s test macro-F1 %.4f (int8 %.4f), k=%d, params=%d", name, metrics["test"]["macro_f1"],
             metrics["test_int8"]["macro_f1"], len(S), metrics["params"])
    return metrics


def student_fidelity(paths: Paths, dataset: str, task: str, teacher_model: str, t_seed: int,
                     model: torch.nn.Module, S: np.ndarray, X_train: np.ndarray, teacher_importance: np.ndarray,
                     fid_cfg: dict, device: torch.device, seed: int) -> dict:
    """Agreement between the student's and teacher's global SHAP rankings on the student's features,
    on a stratified subset of the rows the teacher's SHAP analysis explained."""
    sd = shap_dir(paths, dataset, task, teacher_model, t_seed)
    X_ex, y_ex = np.load(sd / "X_explain.npy"), np.load(sd / "y_explain.npy")
    if len(y_ex) > fid_cfg["n_explain"]:
        sub = stratified_sample(y_ex, fid_cfg["n_explain"], np.random.default_rng(seed), min_per_class=5)
        X_ex, y_ex = X_ex[sub], y_ex[sub]
    bg = np.asarray(X_train[np.load(sd / "background_idx.npy")])[:, S]
    phi = explain(model, bg, X_ex[:, S], y_ex, device, fid_cfg["nsamples"], fid_cfg["batch_size"])
    s_imp, _ = importance_from_phi(phi, y_ex, int(y_ex.max()) + 1)
    t_imp = teacher_importance[S]
    top = max(1, min(5, len(S)))
    overlap = len(set(np.argsort(-s_imp)[:top]) & set(np.argsort(-t_imp)[:top])) / top
    rho = spearmanr(s_imp, t_imp).statistic if len(S) > 1 else 1.0
    return {"spearman": float(np.nan_to_num(rho)), f"top{top}_overlap": overlap, "student_importance": s_imp}
