"""E9: second tiny student = rare-class expert (CICIoT2023 full, 34 classes). Development on VALIDATION only.

Group G = the 12 hard classes chosen by validation in 29_seed_experts (11 low-volume attacks + benign).
Arms (same simple training recipe for every MLP: Adam, class weights count^-0.5, 4 passes over a capped sample):
  A  SHIELD student alone (existing, ~16 KB int8)
  A' same-recipe 1x MLP on 34 classes (recipe-parity check vs SHIELD)
  B  same-recipe 2x MLP on 34 classes (CONTROL: one bigger student, same total size as A + expert)
  C  student + expert MLP (~15k params, ~16 KB int8) trained on G with balanced weights
  D  student + expert MLP distilled from the teacher (ResMLP s0) restricted to G (T=2, 0.5 CE + 0.5 KD)
Combination: p = student mass on G x (lam * expert + (1 - lam) * student-within-G); lam chosen on validation half 1,
reported on validation half 2. Benign false-alarm rate = share of benign rows predicted as any attack.
Decision rule (fixed before the run, applied on the holdout later): keep the expert only if it beats A by >= 1.5
and B by >= 1.0 macro-F1, with benign false alarms not higher than A + 0.1 pt."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import softmax
from shield.eval.predictions import run_dir
from shield.models.train import load_teacher
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("e9")
G_NAMES = ["Backdoor_Malware", "BenignTraffic", "BrowserHijacking", "CommandInjection", "DNS_Spoofing",
           "DictionaryBruteForce", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan", "SqlInjection",
           "Uploading_Attack", "XSS"]


def mlp(d, widths, k):
    layers, a = [], d
    for w in widths:
        layers += [nn.Linear(a, w), nn.ReLU()]; a = w
    return nn.Sequential(*layers, nn.Linear(a, k))


def train(X, y, k, widths, seed, soft=None, epochs=4, T=2.0):
    torch.manual_seed(seed)
    m = mlp(X.shape[1], widths, k)
    cnt = np.maximum(np.bincount(y, minlength=k), 1).astype(float)
    w = cnt ** -0.5; w = w / (w * cnt).sum() * cnt.sum()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    st = torch.tensor(soft, dtype=torch.float32) if soft is not None else None
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(y)), max(1, len(y) // 4096)):
            bi = torch.as_tensor(b); z = m(Xt[bi]); loss = ce(z, yt[bi])
            if st is not None:
                loss = 0.5 * loss + 0.5 * T * T * nn.functional.kl_div(torch.log_softmax(z / T, 1), st[bi], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def proba(m, X):
    return np.concatenate([torch.softmax(m(torch.tensor(X[i:i + 262144])), 1).numpy() for i in range(0, len(X), 262144)])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    args = parser(__doc__, datasets=False).parse_args()
    torch.set_num_threads(8)
    P = Paths(args.smoke)
    d = load_processed(P.processed("ciciot_full"), "class34")
    tr, va = d.splits["train"], d.splits["validation"]
    cls, n = d.classes, d.n_classes
    G = np.array([cls.index(c) for c in G_NAMES]); gmap = np.full(n, -1); gmap[G] = np.arange(len(G))
    ben = cls.index("BenignTraffic")
    rng = np.random.default_rng(0)
    Xv, yv = np.asarray(va.X, np.float32), va.y
    half = rng.random(len(yv)) < 0.5                                     # half 1: choose lambda; half 2: report
    ps = softmax(np.load(P.v3 / "logits" / "ciciot_full_class34" / "shield_s0_validation.npy").astype(np.float64))

    def f1(p, m):
        return 100 * f1_score(yv[m], p[m].argmax(1), labels=np.arange(n), average="macro", zero_division=0)

    def ben_fa(p, m):
        b = m & (yv == ben)
        return 100 * float((p[b].argmax(1) != ben).mean())

    target_fa = ben_fa(ps, half)                                       # student's benign false-alarm rate (half 1)

    def match_fa(p):
        """Corrected comparison (2026-10-12): scale the benign probability so the benign false-alarm rate on half 1
        does not exceed the student's; choose the smallest such scale. Applied identically to every arm."""
        for lb in np.arange(0, 6.01, 0.25):
            q = p.copy(); q[:, ben] *= np.exp(lb); q /= q.sum(1, keepdims=True)
            if ben_fa(q, half) <= target_fa:
                return q, float(lb)
        return q, float(lb)

    res = {"A_student": {"f1_h2": f1(ps, ~half), "benign_fa_h2": ben_fa(ps, ~half), "target_fa_h1": target_fa}}
    # A' / B: same-recipe single students on all 34 classes (capped stratified sample)
    it = stratified_sample(tr.y, 4_000_000, rng, 2000)
    X34, y34 = np.asarray(tr.X[it], np.float32), tr.y[it]
    for name, widths in (("A1_single_1x", (96, 96)), ("B_single_2x", (136, 136))):
        m = train(X34, y34, n, widths, 0)
        pm, lb = match_fa(proba(m, Xv))
        res[name] = {"params": nparams(m), "benign_bias": lb, "f1_h2": f1(pm, ~half), "benign_fa_h2": ben_fa(pm, ~half)}
        log.info("%s: %s", name, res[name])
    # expert data: rows of G only; benign capped at 2M
    ig = np.flatnonzero(np.isin(tr.y, G))
    b = ig[tr.y[ig] == ben]
    if len(b) > 2_000_000:
        ig = np.sort(np.concatenate([ig[tr.y[ig] != ben], rng.choice(b, 2_000_000, replace=False)]))
    Xg, yg = np.asarray(tr.X[ig], np.float32), gmap[tr.y[ig]]
    tnet, _ = load_teacher(run_dir(P, "ciciot_full", "class34", "teacher", 0), torch.device("cpu"))
    tnet.eval()
    with torch.no_grad():
        zt = np.concatenate([tnet(torch.tensor(Xg[i:i + 65536])).numpy() for i in range(0, len(Xg), 65536)])
    soft = softmax(zt[:, G] / 2.0)
    for name, s in (("C_expert_ce", None), ("D_expert_kd", soft)):
        e = train(Xg, yg, len(G), (96, 96), 0, soft=s)
        pe = proba(e, Xv)
        mass = ps[:, G].sum(1, keepdims=True); cond = ps[:, G] / np.maximum(mass, 1e-12)
        best = None
        for lam in (0.25, 0.5, 0.75, 1.0):
            pm = ps.copy(); pm[:, G] = mass * (lam * pe + (1 - lam) * cond)
            pm, lb = match_fa(pm)                                       # same false-alarm level as the student
            sc = f1(pm, half)
            if best is None or sc > best[0]:
                best = (sc, lam, pm)
        res[name] = {"expert_params": nparams(e), "lambda": best[1], "f1_h1": best[0],
                     "f1_h2": f1(best[2], ~half), "benign_fa_h2": ben_fa(best[2], ~half),
                     "rare_f1_h2": {cls[c]: float(100 * f1_score(yv[~half] == c, best[2][~half].argmax(1) == c)) for c in G if c != ben}}
        log.info("%s: lam %s f1_h2 %.2f benign_fa %.2f", name, best[1], res[name]["f1_h2"], res[name]["benign_fa_h2"])
    res["student_rare_f1_h2"] = {cls[c]: float(100 * f1_score(yv[~half] == c, ps[~half].argmax(1) == c)) for c in G if c != ben}
    log.info("A student f1_h2 %.2f benign_fa %.2f", res["A_student"]["f1_h2"], res["A_student"]["benign_fa_h2"])
    save_json(res, P.v3 / "e9_tiny_expert.json")


if __name__ == "__main__":
    main()
