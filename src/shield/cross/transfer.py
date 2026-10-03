"""Cross-dataset generalisation between CICIoT2023 and CICIoMT2024 (same CIC feature extractor).

For each direction source -> target and task (binary, shared5):
  * zero-shot: models trained on the source are tested on the target test split, with the target
    standardised by the source statistics ("source" norm, the naive deployment), by the target's own
    unlabelled training statistics ("target" norm), or adapted without labels by CORAL ("coral") or
    DANN ("dann", students only); student-teacher agreement is reported for every view;
  * few-shot: SHIELD / KD students are fine-tuned on a small labelled share of the target training
    split, against a student trained from scratch on the same rows (does transfer help?);
  * unseen attacks (binary): target attack families absent from the shared label space (MQTT for
    CICIoMT2024; Mirai, Web, BruteForce for CICIoT2023) - are they still flagged as attacks?
Reference rows: in-domain source test score and the target-trained ("oracle") model's score.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from shield.data.datasets import ProcessedData, load_processed, predict_logits, stratified_sample
from shield.data.preprocess import family_of, shared_class
from shield.eval.metrics import classification_metrics
from shield.kd.trainer import kd_dir
from shield.models.student import StudentMLP
from shield.models.train import load_teacher, teacher_dir
from shield.models.xgb import xgb_logits
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything
from shield.xai.shap_analysis import shap_dir

log = get_logger(__name__)


def convert(X: np.ndarray, feats_from: list[str], meta_from: dict, feats_to: list[str], meta_to: dict,
            chunk: int = 2_000_000) -> np.ndarray:
    """Re-standardise rows stored with one dataset's statistics into another's feature space/statistics.
    Processed X = (signed_log(x) - mean) / std, so signed_log(x) is recovered exactly and re-scaled."""
    idx = np.array([feats_from.index(f) for f in feats_to])
    sf, mf = np.asarray(meta_from["scaler"]["std"])[idx], np.asarray(meta_from["scaler"]["mean"])[idx]
    st, mt = np.asarray(meta_to["scaler"]["std"]), np.asarray(meta_to["scaler"]["mean"])
    out = np.empty((len(X), len(idx)), np.float32)
    for i in range(0, len(X), chunk):
        z = np.asarray(X[i:i + chunk])[:, idx] * sf + mf
        out[i:i + chunk] = (z - mt) / st
    return out


class Model:
    """Uniform predict() over the teacher, XGBoost and students (students use their selected columns)."""

    def __init__(self, name: str, kind: str, obj, cols: np.ndarray | None, n_classes: int, run_dir: Path):
        self.name, self.kind, self.obj, self.cols, self.n_classes, self.run_dir = name, kind, obj, cols, n_classes, run_dir

    def logits(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        if self.kind == "xgboost":
            return xgb_logits(self.run_dir, X, self.n_classes)
        Xs = X if self.cols is None else np.asarray(X)[:, self.cols]
        return predict_logits(self.obj, Xs, device)

    def in_domain(self) -> float | None:
        f = self.run_dir / "metrics.json"
        return load_json(f)["test"]["macro_f1"] if f.exists() else None


def load_model(paths: Paths, name: str, ds: str, task: str, seed: int, teacher_model: str, n_classes: int,
               device: torch.device) -> Model | None:
    if name == "teacher":
        d = teacher_dir(paths, ds, task, teacher_model, seed)
        if not (d / "model.pt").exists():
            return None
        m, _ = load_teacher(d, device)
        return Model(name, "torch", m, None, n_classes, d)
    if name == "xgboost":
        d = paths.outputs / "xgboost" / f"{ds}_{task}_s{seed}"
        return Model(name, "xgboost", None, None, n_classes, d) if (d / "model.json").exists() else None
    d = kd_dir(paths, ds, task, name, seed)
    if not (d / "student.pt").exists():
        return None
    ck = torch.load(d / "student.pt", map_location="cpu", weights_only=False)
    m = StudentMLP(len(ck["selected"]), ck["n_classes"], ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    return Model(name, "torch", m.to(device).eval(), np.asarray(ck["selected"]), n_classes, d)


def finetune(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: dict, device: torch.device, seed: int) -> nn.Module:
    """Plain CE fine-tuning on a small labelled set (all rows held on the device)."""
    torch.manual_seed(seed)
    model = copy.deepcopy(model).to(device).train()
    Xt, yt = torch.as_tensor(X, device=device), torch.as_tensor(y, dtype=torch.long, device=device)
    bs = min(cfg["batch_size"], len(y))
    steps = max(cfg["min_steps"], cfg["epochs"] * math.ceil(len(y) / bs))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(yt), (bs,), device=device, generator=g)
        loss = ce(model(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def _sqrtm(C: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** (-0.5 if inverse else 0.5))) @ V.T


def coral_map(X: np.ndarray, Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """CORAL (Sun et al. 2016) without retraining: whiten target rows with the target covariance and
    re-colour them with the source covariance, so the source model sees source-like second-order
    statistics. X, Xs, Xt are in the source's normalisation; Xt are unlabelled target rows."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    I = np.eye(Xs.shape[1])
    A = _sqrtm(np.cov(Xt, rowvar=False) + eps * I, inverse=True) @ _sqrtm(np.cov(Xs, rowvar=False) + eps * I)
    return ((np.asarray(X, dtype=np.float64) - mt) @ A + ms).astype(np.float32)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def dann_adapt(student: nn.Module, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, cfg: dict,
               device: torch.device, seed: int) -> nn.Module:
    """DANN (Ganin et al. 2016): keep the source classification loss while a discriminator on the
    student's last hidden layer, behind a gradient-reversal layer, makes source and target features
    indistinguishable. Starts from the trained source student; target labels are never used."""
    torch.manual_seed(seed)
    model = copy.deepcopy(student).to(device).train()
    layers = list(model.net)
    features, head = nn.Sequential(*layers[:-1]), layers[-1]
    disc = nn.Sequential(nn.Linear(head.in_features, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=cfg["lr"])
    Xs_t, ys_t = torch.as_tensor(Xs, device=device), torch.as_tensor(ys, dtype=torch.long, device=device)
    Xt_t = torch.as_tensor(np.ascontiguousarray(Xt), device=device)
    bs = min(cfg["batch_size"], len(ys), len(Xt))
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    dom = torch.cat([torch.zeros(bs, device=device), torch.ones(bs, device=device)])
    g = torch.Generator(device=device).manual_seed(seed)
    for step in range(cfg["steps"]):
        lam = 2.0 / (1.0 + math.exp(-10.0 * step / cfg["steps"])) - 1.0
        si = torch.randint(0, len(ys_t), (bs,), device=device, generator=g)
        fs = features(Xs_t[si])
        ft = features(Xt_t[torch.randint(0, len(Xt_t), (bs,), device=device, generator=g)])
        d_logits = disc(_GradReverse.apply(torch.cat([fs, ft]), lam)).squeeze(1)
        loss = ce(head(fs), ys_t[si]) + bce(d_logits, dom)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def _agreement(logits: np.ndarray, teacher_pred: np.ndarray | None) -> dict:
    return {} if teacher_pred is None else {"agreement_teacher": float((logits.argmax(1) == teacher_pred).mean())}


def _metrics(y: np.ndarray, logits: np.ndarray, data: ProcessedData) -> dict:
    m = classification_metrics(y, logits, data.n_classes, data.classes, full=True)
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "mcc": m["mcc"],
            "recall": {c: v["recall"] for c, v in m["per_class"].items()}}


def unseen_attacks(paths: Paths, cfg: dict, src: ProcessedData, tgt: str, models: list[Model],
                   device: torch.device) -> list[dict]:
    """Binary detectors on target attack families that have no counterpart in the shared label space."""
    spec = cfg["unseen"].get(tgt)
    if not spec or not (paths.processed(spec["base"]) / "meta.json").exists():
        return []
    base = load_processed(paths.processed(spec["base"]), spec["task"], splits=("test",))
    source = base.meta["source"]
    te = base.splits["test"]
    names = np.asarray(base.classes)[te.y]
    unseen = np.array([shared_class(source, n) is None for n in names])
    if not unseen.any():
        return []
    Xu = np.asarray(te.X[unseen])
    fam = np.array([family_of(source, n) for n in names[unseen]])
    tgt_meta = load_json(paths.processed(tgt) / "meta.json")
    views = {"source": convert(Xu, base.features, base.meta, src.features, src.meta),
             "target": convert(Xu, base.features, base.meta, src.features, tgt_meta)}
    attack_idx = src.classes.index("Attack")
    rows = []
    for m in models:
        for norm, X in views.items():
            pred = m.logits(X, device).argmax(1) == attack_idx
            for f in np.unique(fam):
                sel = fam == f
                rows.append({"model": m.name, "norm": norm, "family": str(f), "rows": int(sel.sum()),
                             "detection_rate": float(pred[sel].mean())})
    return rows


def run_block(paths: Paths, cfg: dict, src: str, tgt: str, task: str, seed: int, teacher_model: str,
              device: torch.device, force: bool = False) -> dict:
    out = paths.out("cross") / f"{src}_to_{tgt}_{task}_s{seed}.json"
    if out.exists() and not force:
        return load_json(out)
    seed_everything(seed)
    S = load_processed(paths.processed(src), task, splits=("train", "test"))
    T = load_processed(paths.processed(tgt), task, splits=("train", "test"))
    if S.features != T.features or S.classes != T.classes:
        raise ValueError(f"{src} and {tgt} must share features and classes for task {task}")
    te, tr, s_tr = T.splits["test"], T.splits["train"], S.splits["train"]
    rng = np.random.default_rng(seed)
    # Unlabelled target training rows in the source's normalisation (for CORAL / DANN; labels unused).
    t_idx = np.sort(rng.choice(len(tr.y), min(len(tr.y), max(cfg["coral"]["sample_rows"], cfg["dann"]["target_rows"])),
                               replace=False))
    Xt_unlab = convert(tr.X[t_idx], T.features, T.meta, S.features, S.meta)
    views = {"target": np.asarray(te.X), "source": convert(te.X, T.features, T.meta, S.features, S.meta)}
    s_idx = np.sort(rng.choice(len(s_tr.y), min(len(s_tr.y), cfg["coral"]["sample_rows"]), replace=False))
    views["coral"] = coral_map(views["source"], np.asarray(s_tr.X[s_idx]),
                               Xt_unlab[:cfg["coral"]["sample_rows"]], cfg["coral"]["eps"])
    res = {"source": src, "target": tgt, "task": task, "seed": seed, "zero_shot": [], "fewshot": [], "unseen": []}

    models, teacher_pred = [], {}
    for name in cfg["zero_shot_models"]:
        m = load_model(paths, name, src, task, seed if name != "xgboost" else 0, teacher_model, S.n_classes, device)
        if m is None:
            log.warning("missing %s for %s/%s seed %d", name, src, task, seed)
            continue
        models.append(m)
        oracle = load_model(paths, name, tgt, task, seed if name != "xgboost" else 0, teacher_model,
                            T.n_classes, device)
        for norm, X in views.items():
            logits = m.logits(X, device)
            if name == "teacher":
                teacher_pred[norm] = logits.argmax(1)
            res["zero_shot"].append({"model": name, "norm": norm, "in_domain_macro_f1": m.in_domain(),
                                     "oracle_macro_f1": oracle.in_domain() if oracle else None,
                                     **_agreement(logits, teacher_pred.get(norm)), **_metrics(te.y, logits, T)})

    # DANN: fine-tune the source student with a domain discriminator on unlabelled target rows.
    dcfg = cfg["dann"]
    ds_idx = stratified_sample(s_tr.y, min(len(s_tr.y), dcfg["source_rows"]), rng, min_per_class=5)
    Xs_lab, ys_lab = np.asarray(s_tr.X[ds_idx]), s_tr.y[ds_idx]
    for m in models:
        if m.name not in dcfg["models"]:
            continue
        adapted = dann_adapt(m.obj, Xs_lab[:, m.cols], ys_lab, Xt_unlab[:dcfg["target_rows"]][:, m.cols], dcfg,
                             device, seed)
        logits = predict_logits(adapted, views["source"][:, m.cols], device)
        res["zero_shot"].append({"model": m.name, "norm": "dann", "in_domain_macro_f1": m.in_domain(),
                                 "oracle_macro_f1": None, **_agreement(logits, teacher_pred.get("source")),
                                 **_metrics(te.y, logits, T)})
    students = [m for m in models if m.name in cfg["fewshot_models"]]
    for frac in cfg["fractions"]:
        n = max(int(frac * len(tr.y)), 5 * T.n_classes)
        idx = stratified_sample(tr.y, n, rng, min_per_class=5)
        Xf, yf = np.asarray(tr.X[idx]), tr.y[idx]
        for m in students:
            hidden = [l.out_features for l in m.obj.net if isinstance(l, nn.Linear)][:-1]
            tuned = finetune(m.obj, Xf[:, m.cols], yf, cfg["finetune"], device, seed)
            # Same architecture, same rows, no source knowledge: does transferring help?
            scratch = finetune(StudentMLP(len(m.cols), T.n_classes, hidden), Xf[:, m.cols], yf, cfg["finetune"],
                               device, seed)
            for kind, model in (("finetuned", tuned), ("scratch", scratch)):
                logits = predict_logits(model, views["target"][:, m.cols], device)
                res["fewshot"].append({"model": m.name, "kind": kind, "fraction": frac,
                                       "labelled_rows": int(len(idx)), **_metrics(te.y, logits, T)})

    if task == "binary":
        res["unseen"] = unseen_attacks(paths, cfg, S, tgt, models, device)
    save_json(res, out)
    zs = {(r["model"], r["norm"]): r["macro_f1"] for r in res["zero_shot"]}
    log.info("%s -> %s %s s%d zero-shot macro-F1: %s", src, tgt, task, seed,
             {f"{k[0]}/{k[1]}": round(v, 3) for k, v in zs.items()})
    return res


def shap_agreement(paths: Paths, pairs: list, tasks: list[str], teacher_model: str, seed: int = 0) -> dict:
    """Spearman correlation of the two datasets' teacher SHAP importances over the shared features."""
    out = {}
    for a, b in pairs:
        for task in tasks:
            fa = shap_dir(paths, a, task, teacher_model, seed) / "importance.json"
            fb = shap_dir(paths, b, task, teacher_model, seed) / "importance.json"
            if fa.exists() and fb.exists():
                ia, ib = load_json(fa), load_json(fb)
                rho = spearmanr(ia["importance"], ib["importance"]).statistic
                top = lambda d: set(d["ranked_features"][:10])
                out[f"{a}|{b}|{task}"] = {"spearman": float(rho), "top10_overlap": len(top(ia) & top(ib)) / 10,
                                          "top10_" + a: ia["ranked_features"][:10], "top10_" + b: ib["ranked_features"][:10]}
    return out
