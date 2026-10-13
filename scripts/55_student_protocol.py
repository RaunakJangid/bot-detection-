"""P1: the tiny student under published-style vs strict protocols (companion to scripts/39_protocol_ablation.py).

Same cells as E4 (IAT kept/removed x dedup on/off; CICIoMT also official vs random split; CICIoT always random).
In every cell: the cell's own XGBoost teacher (as in E4: depth 10, lr 0.1, <= 2000 trees, early stopping, rows capped
at --cap) is distilled into the SHIELD student recipe chosen in v3 tuning (outputs_full/tuning/kd_choice.json):
top-k features by the teacher's gain importance (k = 32 for 34-class / attack, 16 for category), MLP 128-64, KD loss
alpha * KL(T=1) + (1 - alpha) * CE with alpha 0.7, signed-log + z-score fitted on train, Adam 1e-3, batch 4096,
<= 30 epochs with early stopping on validation macro-F1 (patience 4). 3 seeds. Test scored once per cell and seed.
Reported: student and teacher test macro-F1 / accuracy per cell; inflation = published-style cell minus strict cell."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from shield.cli import parser
from shield.data.datasets import stratified_sample
from shield.data.preprocess import LABEL, cic_task_specs, dedup_across_splits, row_hash
from shield.models.student import StudentMLP
from shield.utils.config import Paths, load_config
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("student_protocol")
_s = importlib.util.spec_from_file_location("e4", Path(__file__).with_name("39_protocol_ablation.py"))
e4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e4)
JOBS = [("ciciot", "class34", 32), ("ciciomt", "category", 16), ("ciciomt", "attack", 32)]
SEEDS = (0, 1, 2)


def slog(X):
    return np.sign(X) * np.log1p(np.abs(X))


def train_student(Xtr, ytr, Ptr, Xv, yv, n, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    m = StudentMLP(Xtr.shape[1], n, [128, 64]).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, yt, Pt = (torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev), torch.tensor(Ptr, device=dev))
    Xvt = torch.tensor(Xv, device=dev)
    best, best_state, bad = -1, None, 0
    g = torch.Generator(device=dev); g.manual_seed(seed)
    for ep in range(30):
        m.train()
        perm = torch.randperm(len(Xt), device=dev, generator=g)
        for i in range(0, len(Xt), 4096):
            b = perm[i:i + 4096]
            lo = m(Xt[b])
            loss = 0.7 * F.kl_div(F.log_softmax(lo, 1), Pt[b], reduction="batchmean") + 0.3 * F.cross_entropy(lo, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.cat([m(Xvt[i:i + 200_000]).argmax(1) for i in range(0, len(Xvt), 200_000)]).cpu().numpy()
        f = f1_score(yv, pv, average="macro")
        if f > best:
            best, bad = f, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(best_state)
    return m.eval()


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--cap", type=int, default=2_000_000)
    args = p.parse_args()
    P, cfg = Paths(args.smoke), load_config("data", args.smoke)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_f = P.v3 / "ablation" / "student_protocol.json"
    res = load_json(out_f) if out_f.exists() else {}
    for src in ("ciciot", "ciciomt"):
        for split in (("official", "random") if src == "ciciomt" else ("official",)):
            fr = e4.frames_for(src, P, cfg, split)
            labels = sorted(set().union(*(set(df[LABEL].unique()) for df in fr.values())))
            specs = cic_task_specs(src, labels, False)
            all_feats = [c for c in fr["train"].columns if c != LABEL]
            for iat, dedup in itertools.product((False, True), (True, False)):
                feats = [c for c in all_feats if iat or c != "IAT"]
                lab_index = {l: i for i, l in enumerate(labels)}
                codes = {s: df[LABEL].map(lab_index).to_numpy(np.int64) for s, df in fr.items()}
                keep = (dedup_across_splits({s: row_hash(df[feats]) for s, df in fr.items()}, codes)[0] if dedup
                        else {s: np.ones(len(df), bool) for s, df in fr.items()})
                X = {s: fr[s][feats].to_numpy(np.float32)[keep[s]] for s in fr}
                for s in X:
                    X[s][~np.isfinite(X[s])] = 0
                for _, task, k in [j for j in JOBS if j[0] == src]:
                    cell = f"{src}_{task}|split={split}|IAT={'kept' if iat else 'removed'}|dedup={'on' if dedup else 'off'}"
                    if all(f"{cell}|student|s{s}" in res for s in SEEDS):
                        continue
                    classes, fn = specs[task]
                    ci = {c: i for i, c in enumerate(classes)}
                    lut = np.array([ci[fn(l)] for l in labels])
                    y = {s: lut[codes[s][keep[s]]] for s in fr}
                    n = len(classes)
                    rng = np.random.default_rng(0)
                    it = stratified_sample(y["train"], min(args.cap, len(y["train"])), rng, 1000)
                    iv = stratified_sample(y["validation"], min(500_000, len(y["validation"])), rng, 200)
                    tch = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist",
                                            device="cuda", random_state=0, early_stopping_rounds=50, subsample=0.8,
                                            colsample_bytree=0.8, objective="multi:softprob", num_class=n)
                    tch.fit(X["train"][it], y["train"][it], eval_set=[(X["validation"][iv], y["validation"][iv])], verbose=False)
                    proba = lambda A: np.vstack([tch.predict_proba(A[i:i + 500_000]) for i in range(0, len(A), 500_000)])
                    Pt = proba(X["test"]); pt = Pt.argmax(1)
                    res[f"{cell}|teacher"] = {"macro_f1": float(100 * f1_score(y["test"], pt, average="macro")),
                                              "accuracy": float(100 * accuracy_score(y["test"], pt))}
                    top = np.argsort(-tch.feature_importances_)[:k]
                    Ptr = proba(X["train"][it]).astype(np.float32)
                    Ztr = slog(X["train"][it][:, top]); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
                    z = lambda A: ((slog(A[:, top]) - mu) / sd).astype(np.float32)
                    Zv, Zt = z(X["validation"][iv]), z(X["test"])
                    for s in SEEDS:
                        m = train_student((Ztr - mu) / sd, y["train"][it], Ptr, Zv, y["validation"][iv], n, s, dev)
                        with torch.no_grad():
                            Ztt = torch.tensor(Zt, device=dev)
                            ps = torch.cat([m(Ztt[i:i + 200_000]).argmax(1) for i in range(0, len(Ztt), 200_000)]).cpu().numpy()
                        res[f"{cell}|student|s{s}"] = {"macro_f1": float(100 * f1_score(y["test"], ps, average="macro")),
                                                       "accuracy": float(100 * accuracy_score(y["test"], ps)),
                                                       "params": int(sum(q.numel() for q in m.parameters())), "k": k}
                        log.info("%s | teacher %.2f | student s%d %.2f (acc %.2f)", cell, res[f"{cell}|teacher"]["macro_f1"],
                                 s, res[f"{cell}|student|s{s}"]["macro_f1"], res[f"{cell}|student|s{s}"]["accuracy"])
                        save_json(res, out_f)


if __name__ == "__main__":
    main()
