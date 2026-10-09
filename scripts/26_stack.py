"""Stacking on VALIDATION only: a multinomial logistic regression (or a 1-hidden-layer MLP) over the concatenated
log-probabilities of all available member models. Settings are chosen by 2-fold cross-fitting on validation
(out-of-fold macro-F1, then the validation-fitted per-class bias on the out-of-fold scores); the final stacker
is refit on all validation rows and applied to test once. Members = every <model>_s<seed> in the logits cache
that has validation and test files (deep baselines excluded with --exclude)."""

import json

import numpy as np
import torch
from torch import nn

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.utils.config import Paths
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("stack")


def train(Xf, y, n, hidden, wd, cw, device, epochs=30, seed=0):
    torch.manual_seed(seed)
    d = Xf.shape[1]
    net = (nn.Linear(d, n) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    X, Y = torch.as_tensor(Xf, device=device), torch.as_tensor(y, device=device)
    w = torch.as_tensor(cw, dtype=torch.float32, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(Y), device=device)
        for i in range(0, len(Y), 8192):
            b = perm[i:i + 8192]
            loss = nn.functional.cross_entropy(net(X[b]), Y[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def predict(net, Xf, device):
    return torch.cat([net(torch.as_tensor(Xf[i:i + 262144], device=device)).cpu() for i in range(0, len(Xf), 262144)]).numpy()


def main():
    p = parser(__doc__)
    p.add_argument("--exclude", nargs="*", default=["cnn_lstm", "cnn_gru", "ftt", "ens4", "two_stage", "ensemble"])
    args = p.parse_args()
    paths, device = Paths(args.smoke), get_device(args.cpu)
    ds, task = args.dataset, args.task
    data = load_processed(paths.processed(ds), task, splits=("train", "validation", "test"))
    yv, yt, n = data.splits["validation"].y, data.splits["test"].y, data.n_classes
    prior = np.bincount(data.splits["train"].y, minlength=n) / len(data.splits["train"].y)
    lc = paths.v3 / "logits" / f"{ds}_{task}"
    members = sorted({f.name[:-len("_validation.npy")] for f in lc.glob("*_validation.npy")
                      if (lc / f.name.replace("_validation", "_test")).exists()
                      and not any(f.name.startswith(e + "_") for e in args.exclude)})
    log.info("%d members: %s", len(members), members)
    lsm = lambda z: z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    F = {s: np.concatenate([np.clip(lsm(np.load(lc / f"{m}_{s}.npy").astype(np.float32)), -20, 0) for m in members], 1)
         for s in ("validation", "test")}
    cnt = np.bincount(yv, minlength=n).astype(float)
    cw = 1 / np.sqrt(np.maximum(cnt, 1)); cw = cw / (cw * cnt).sum() * cnt.sum()
    rng = np.random.default_rng(0)
    fold = np.zeros(len(yv), int)
    fold[stratified_sample(yv, len(yv) // 2, rng, 1)] = 1
    best = None
    for hidden in (0, 128):
        for wd in (1e-4, 1e-2):
            oof = np.zeros((len(yv), n), np.float32)
            for k in (0, 1):
                net = train(F["validation"][fold != k], yv[fold != k], n, hidden, wd, cw, device)
                oof[fold == k] = predict(net, F["validation"][fold == k], device)
            raw = classification_metrics(yv, oof, n)["macro_f1"]
            rule = fit_decision_rule(oof, yv, prior, device)
            log.info("%s %s stack hidden=%d wd=%g: OOF val macro-F1 %.4f (+rule %.4f)", ds, task, hidden, wd, raw,
                     rule["val_macro_f1_bias"])
            if best is None or rule["val_macro_f1_bias"] > best[0]:
                best = (rule["val_macro_f1_bias"], hidden, wd, rule["bias"], raw)
    _, hidden, wd, bias, raw = best
    net = train(F["validation"], yv, n, hidden, wd, cw, device)
    zt = predict(net, F["test"], device)
    m_raw = classification_metrics(yt, zt, n, data.classes, full=True)
    m_rule = classification_metrics(yt, apply_bias(zt, bias), n, data.classes, full=True)
    np.save(lc / "stack_s0_test.npy", apply_bias(zt, bias).astype(np.float16))
    ref = json.load(open(paths.v3 / "detector_v4" / f"{ds}_{task}.json"))
    ens = {r["model"]: r["test_macro_f1"] for r in ref["rows"] if r["model"].startswith("ens4")}
    save_json({"members": members, "hidden": hidden, "weight_decay": wd, "oof_val_macro_f1": raw, "oof_val_rule": best[0],
               "test": m_raw, "test_rule": m_rule, "ens4_test": ens}, paths.v3 / "stack" / f"{ds}_{task}.json")
    log.info("%s %s STACK (hidden=%d wd=%g): test macro-F1 %.4f, +rule %.4f | ens4 %s", ds, task, hidden, wd,
             m_raw["macro_f1"], m_rule["macro_f1"], {k: round(v, 4) for k, v in ens.items()})


if __name__ == "__main__":
    main()
