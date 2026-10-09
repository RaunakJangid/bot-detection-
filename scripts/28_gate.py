"""Learned escalation gate vs confidence (and 'always escalate benign'). Gate = logistic regression on the
student's log-probs, trained on VALIDATION half A to predict 'student wrong and expert right'; thresholds per
budget set on half B; test scored once. Student = SHIELD s0 (+rule), expert = specialist_auto (else ens4+rule)."""
import json
import numpy as np
import torch
from torch import nn
from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics, softmax
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.utils.config import Paths
from shield.utils.io import save_json

a = parser(__doc__).parse_args()
P, ds, task = Paths(), a.dataset, a.task
d = load_processed(P.processed(ds), task)
n, yv, yt = d.n_classes, d.splits["validation"].y, d.splits["test"].y
prior = np.bincount(d.splits["train"].y, minlength=n) / len(d.splits["train"].y)
lc = P.v3 / "logits" / f"{ds}_{task}"
ld = lambda m, s: np.load(lc / f"{m}_{s}.npy").astype(np.float32)
sv, st = ld("shield_s0", "validation"), ld("shield_s0", "test")
b = fit_decision_rule(sv, yv, prior)["bias"]
sv, st = apply_bias(sv, b), apply_bias(st, b)
eb = np.asarray(json.load(open(P.v3 / "detector_v4" / f"{ds}_{task}.json"))["rule_bias"])
ev, et = apply_bias(ld("ens4_s0", "validation"), eb), apply_bias(ld("ens4_s0", "test"), eb)
spec = lc / "specialist_auto_s0_test.npy"
if spec.exists():   # the specialist only stores test outputs; use it for the test expert when available
    et = np.load(spec).astype(np.float32)
ben = next(i for i, c in enumerate(d.classes) if c in ("BenignTraffic", "Benign", "Normal"))
f1 = lambda y, p: classification_metrics(y, np.eye(n, dtype=np.float32)[p], n)["macro_f1"]
A = np.zeros(len(yv), bool); A[stratified_sample(yv, len(yv) // 2, np.random.default_rng(0), 1)] = True
feat = lambda s: np.concatenate([s, softmax(s).max(1, keepdims=True)], 1)
target = ((sv.argmax(1) != yv) & (ev.argmax(1) == yv)).astype(np.float32)
X = torch.as_tensor(feat(sv[A])); Y = torch.as_tensor(target[A])
g = nn.Linear(X.shape[1], 1); opt = torch.optim.Adam(g.parameters(), 1e-2)
for _ in range(300):
    opt.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(g(X).squeeze(1), Y); l.backward(); opt.step()
score = lambda s: g(torch.as_tensor(feat(s))).squeeze(1).detach().numpy()
gv, gt = score(sv[~A]), score(st)
cv, ct = -softmax(sv[~A]).max(1), -softmax(st).max(1)   # higher = escalate
out = {}
for name, (sb, stest) in {"confidence": (cv, ct), "gate": (gv, gt)}.items():
    for benign in (False, True):
        for bud in (0.05, 0.1, 0.2):
            thr = np.quantile(sb, 1 - bud)
            esc = stest > thr
            if benign:
                esc |= st.argmax(1) == ben
            pred = np.where(esc, et.argmax(1), st.argmax(1))
            k = f"{name}{'+benign' if benign else ''}@{bud}"
            out[k] = {"test_f1": f1(yt, pred), "escalated": float(esc.mean())}
            print(k, round(100 * out[k]["test_f1"], 2), round(100 * out[k]["escalated"], 1), flush=True)
save_json(out, P.v3 / "gate" / f"{ds}_{task}.json")
