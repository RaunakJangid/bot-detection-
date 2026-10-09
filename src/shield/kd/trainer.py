"""One distillation run = (dataset, task, variant spec, seed).

A run's settings are resolved in this order (later wins):
    kd.yaml defaults -> validation-tuned choice (outputs/tuning/kd_choice.json, if use_tuning)
    -> the variant's own overrides -> an explicit k (F1-vs-k sweep).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from shield.data.datasets import Batcher, class_weights, load_processed, predict_logits, stratified_sample
from shield.data.registry import Registry
from shield.engine import fit
from shield.eval.metrics import classification_metrics, confusion
from shield.kd.cache import cache_dir, load_cache_dir, write_cache
from shield.kd.losses import ShieldLoss
from shield.models.student import StudentMLP, count_params, quantize_int8
from shield.models.train import train_cfg_for
from shield.utils.config import Paths
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything
from shield.xai.selection import select_features
from shield.xai.shap_analysis import explain, importance_from_phi, shap_dir

log = get_logger(__name__)

SPEC_KEYS = ("select", "ranking", "kd", "T", "alpha", "beta", "gamma", "dkd_alpha", "dkd_beta", "teacher",
             "k", "hidden", "train_fraction")


def kd_dir(paths: Paths, dataset: str, task: str, variant: str, seed: int, root: str = "kd",
           base: Path | None = None) -> Path:
    return (base or paths.outputs) / root / f"{dataset}_{task}" / f"{variant}_s{seed}"


def ensemble_cache_dir(paths: Paths, dataset: str, task: str) -> Path:
    """v3 step 1: soft targets of the validation-weighted teacher ensemble (11_ensemble_teacher.py)."""
    return paths.v3 / "teacher_ens" / f"{dataset}_{task}" / "cache"


def tuning_choice(paths: Paths) -> dict | None:
    f = paths.outputs / "tuning" / "kd_choice.json"
    return load_json(f) if f.exists() else None


def resolve_spec(cfg: dict, variant: dict, dataset: str, task: str, choice: dict | None,
                 k_override: int | None = None) -> dict:
    loss = cfg["loss"]
    spec = {"select": "shap", "ranking": "global", "teacher": "teacher", "k": cfg["k"],
            "hidden": list(cfg["student"]["hidden"]), "train_fraction": None,
            **{key: loss[key] for key in ("kd", "T", "alpha", "beta", "gamma", "dkd_alpha", "dkd_beta")}}
    if choice:
        spec.update(choice["objective"])
        spec["ranking"] = choice.get("ranking", spec["ranking"])
        reg = Registry()
        if not reg.spec(dataset).get("shared"):  # per-task size/feature budget (shared datasets keep defaults)
            src = {"ciciot_full": "ciciot"}.get(reg.source(dataset), reg.source(dataset))   # v3: reuse the
            spec.update(choice.get("per_task", {}).get(f"{src}_{task}", {}))   # subset's validation choice
    spec.update({key: variant[key] for key in SPEC_KEYS if key in variant})
    if k_override is not None:
        spec["k"] = int(k_override)
    return spec


def class_balanced_importance(info: dict) -> np.ndarray:
    """Mean over classes of each class's normalised |SHAP| profile, so rare classes count as much as DDoS."""
    pc = np.asarray(info["per_class_importance"], dtype=np.float64)
    pc = pc[pc.sum(1) > 0]
    return (pc / pc.sum(1, keepdims=True)).mean(0)


def _columns(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    return X if len(S) == X.shape[1] else np.asarray(X[:, S])


def run_distillation(paths: Paths, dataset: str, task: str, variant: str, seed: int, cfg: dict,
                     teacher_model: str, device: torch.device, force: bool = False,
                     k_override: int | None = None, variant_spec: dict | None = None, run_name: str | None = None,
                     root: str = "kd", use_tuning: bool | None = None, make_cache: bool = False,
                     v3: bool = False) -> dict:
    """variant: a key of cfg["variants"] (or a label when `variant_spec` is given explicitly).
    k_override stores the run as '<variant>_k<k>'. make_cache=True (teacher assistant) also caches this
    model's outputs for other students to distil from. v3=True writes under the v3 root."""
    run_name = run_name or (variant if k_override is None else f"{variant}_k{k_override}")
    out = kd_dir(paths, dataset, task, run_name, seed, root, base=paths.v3 if v3 else None)
    if (out / "metrics.json").exists() and not force and (not make_cache or (out / "cache" / "done.json").exists()):
        log.info("KD %s/%s exists, skipping", out.parent.name, out.name)
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    use_tuning = cfg.get("use_tuning", False) if use_tuning is None else use_tuning
    choice = tuning_choice(paths) if use_tuning else None
    v = variant_spec if variant_spec is not None else cfg["variants"][variant]
    spec = resolve_spec(cfg, v, dataset, task, choice, k_override)
    tcfg = train_cfg_for(cfg, dataset)

    data = load_processed(paths.processed(dataset), task)
    tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
    t_seed = cfg["teacher_seed"]
    shap_info = load_json(shap_dir(paths, dataset, task, teacher_model, t_seed) / "importance.json")
    global_imp = np.asarray(shap_info["importance"], dtype=np.float64)
    importance = class_balanced_importance(shap_info) if spec["ranking"] == "class_balanced" else global_imp
    k = shap_info["k95"] if spec["k"] == "k95" else int(spec["k"])

    rng = np.random.default_rng(seed)
    sample_idx = stratified_sample(tr.y, min(cfg["mi_sample"], len(tr.y)), rng, min_per_class=10)
    S = select_features(spec["select"], k, data.n_features, shap_ranking=np.argsort(-importance),
                        X_sample=np.asarray(tr.X[sample_idx]), y_sample=tr.y[sample_idx],
                        meta=data.meta, seed=seed)
    w = importance[S] / max(importance[S].max(), 1e-12)

    loss = ShieldLoss(spec["alpha"], spec["beta"], spec["gamma"], spec["T"],
                      feature_weights=torch.as_tensor(w, dtype=torch.float32, device=device),
                      kd=spec["kd"], dkd_alpha=spec["dkd_alpha"], dkd_beta=spec["dkd_beta"])
    cw = class_weights(tr.y, data.n_classes, tcfg["class_weight"])
    if cw is not None:
        loss.ce.weight = torch.as_tensor(cw, device=device)

    # Which model the student distils from, and the original teacher's test predictions (agreement).
    teacher_cache = cache_dir(paths, dataset, task, teacher_model, t_seed)
    kd_cache = {"assistant": kd_dir(paths, dataset, task, "assistant", t_seed, root) / "cache",
                "ensemble": ensemble_cache_dir(paths, dataset, task)}.get(spec["teacher"], teacher_cache)
    _, _, teacher_pred = load_cache_dir(teacher_cache)
    extras = []
    if loss.needs_logits or loss.needs_attr:
        t_logits, t_attr, kd_pred = load_cache_dir(kd_cache)
        extras.append(t_logits)
        if loss.needs_attr:
            if t_attr is None:
                raise ValueError(f"{kd_cache} has no attributions; the attribution loss needs gamma = 0 here")
            extras.append(t_attr)
    else:
        kd_pred = teacher_pred

    # Rows used for training (all, or a stratified fraction for the low-data experiment).
    rows = None
    if spec["train_fraction"]:
        rows = stratified_sample(tr.y, max(int(spec["train_fraction"] * len(tr.y)), 5 * data.n_classes),
                                 np.random.default_rng(1000 + seed), min_per_class=5)
        tcfg = {**tcfg, "samples_per_epoch": max(len(rows), cfg["lowdata"]["min_epoch_samples"])}
    X_tr = _columns(tr.X, S) if rows is None else np.asarray(tr.X[rows])[:, S]
    y_tr = tr.y if rows is None else tr.y[rows]
    extras = [e if rows is None else np.asarray(e[rows]) for e in extras]
    if loss.needs_attr:
        extras[-1] = _columns(extras[-1], S)

    model = StudentMLP(len(S), data.n_classes, spec["hidden"])
    batcher = Batcher(X_tr, y_tr, tcfg["batch_size"], device, balance_power=tcfg["balance_power"],
                      samples_per_epoch=tcfg["samples_per_epoch"], extras=extras, seed=seed)
    name = f"{dataset}_{task}/{run_name}_s{seed}"
    t0 = time.time()
    state, history = fit(model, batcher, _columns(va.X, S), va.y, loss, tcfg, device, data.n_classes, name)
    train_seconds = time.time() - t0
    del batcher, extras, X_tr

    X_te = _columns(te.X, S)
    logits = predict_logits(model, X_te, device, tcfg["eval_batch_size"])
    np.save(out / "confusion.npy", confusion(te.y, logits, data.n_classes))
    q_model = quantize_int8(model)
    q_logits = predict_logits(q_model, X_te, torch.device("cpu"), tcfg["eval_batch_size"])
    pred, q_pred = logits.argmax(1), q_logits.argmax(1)

    torch.save({"state_dict": state, "selected": S, "features": [data.features[i] for i in S],
                "hidden": spec["hidden"], "n_classes": data.n_classes, "classes": data.classes, "spec": spec},
               out / "student.pt")
    torch.save(q_model, out / "student_int8.pt")
    if make_cache:  # teacher assistant: cache it like a teacher (it uses all features, so S = all)
        write_cache(model, _columns(tr.X, S), X_te, data.n_classes, out / "cache", device)

    metrics = {
        "dataset": dataset, "task": task, "variant": variant, "run": run_name, "seed": seed, "k": int(len(S)),
        "selection": spec["select"], "spec": spec, "loss": spec, "selected": S,
        "selected_features": [data.features[i] for i in S], "params": count_params(model),
        "train_rows": int(len(y_tr)), "train_seconds": round(train_seconds, 1),
        "best_val_macro_f1": max(h["val_macro_f1"] for h in history),
        "test": classification_metrics(te.y, logits, data.n_classes, data.classes, full=True),
        "test_int8": classification_metrics(te.y, q_logits, data.n_classes, data.classes, full=True),
        "agreement": {"teacher": float((pred == teacher_pred).mean()), "kd_teacher": float((pred == kd_pred).mean()),
                      "teacher_int8": float((q_pred == teacher_pred).mean())},
        "history": history, "hardware": hardware_record(),
    }
    fid = cfg["fidelity"]
    metrics["fidelity"] = (student_fidelity(paths, dataset, task, teacher_model, t_seed, model, S, tr.X,
                                            global_imp, fid, device, seed)
                           if seed in fid["seeds"] and root == "kd" and not v3 else None)
    save_json(metrics, out / "metrics.json")
    log.info("KD %s test macro-F1 %.4f (int8 %.4f), agreement %.4f, k=%d, params=%d", name,
             metrics["test"]["macro_f1"], metrics["test_int8"]["macro_f1"], metrics["agreement"]["teacher"],
             len(S), metrics["params"])
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
