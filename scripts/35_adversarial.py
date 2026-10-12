"""Realistic (problem-space-constrained) evasion of the detectors on CICIoT2023 34-class, no leakage.

Threat model (after Apruzzese et al., DTRAP 2022; Chernikova & Oprea, FENCE, ACM TOPS 2022; Sheatsley et al.):
  * the attacker controls only its own (malicious) flows and wants them labelled Benign;
  * it can only (a) PAD packets, scaling all packet-size features by s >= 1, and (b) change its SENDING RATE,
    scaling time by t (t > 1 = slower: flow_duration * t, Rate / Srate / t);
  * derived features stay consistent with these two knobs: sizes (Tot sum, Min, Max, AVG, Tot size, Std,
    Radius) * s, Magnitue * sqrt(s), Covariance * s^2; flags, counts, protocol indicators, TTL, header length,
    the in/out variance ratio and Weight are unchanged (padding and pacing do not alter them);
  * black-box, query-based: per flow the attacker tries every (s, t) on a grid within its budget (<= 64
    queries) and succeeds if any is classified Benign. Identical procedure for every model, so trees, the
    integer model and neural nets are compared fairly (no gradient-masking issues).
Budget: s <= S (padding cost), t <= T (attack slowed T-fold: a DoS slowed 10x is 10x weaker). Reported:
evasion rate = share of attack flows that the model detects (not Benign) when unperturbed and that become
Benign under some allowed (s, t); per family as well. Raw features are recovered by inverting the strict
preprocessing (signed-log + standardisation fitted on train) and re-applied after perturbation."""

import importlib.util
import json
import os

import numpy as np

from shield.cli import parser
from shield.data.datasets import load_processed, predict_logits, stratified_sample
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("adv")
S_GRID = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
T_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
SIZE = ["Tot sum", "Min", "Max", "AVG", "Tot size", "Std", "Radius"]
TIME_MUL, TIME_DIV = ["flow_duration"], ["Rate", "Srate"]


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--dataset", default="ciciot_full")
    p.add_argument("--task", default="class34")
    p.add_argument("--n", type=int, default=20_000)
    args = p.parse_args()
    if args.dataset == "ciciot_full":
        os.environ["SHIELD_OUTPUTS_DIR"] = r"D:\shield_run\outputs_full"
    from shield.eval.metrics import classification_metrics
    from shield.eval.predictions import load_student, run_dir
    from shield.models.train import load_teacher
    from shield.models.xgb import load_xgb, xgb_logits
    from shield.utils.config import Paths
    from shield.utils.device import get_device
    P, dev = Paths(args.smoke), get_device(args.cpu)
    ds, task = args.dataset, args.task
    d = load_processed(P.processed(ds), task)
    meta = json.load(open(P.processed(ds) / "meta.json"))
    sc = {k: np.asarray(v, np.float32) for k, v in meta["scaler"].items() if k != "transform"}
    F = d.features
    ix = {f: F.index(f) for f in F}
    ben = d.classes.index("BenignTraffic") if "BenignTraffic" in d.classes else d.classes.index("Benign")
    te, va, tr = d.splits["test"], d.splits["validation"], d.splits["train"]
    rng = np.random.default_rng(0)
    att = np.flatnonzero(te.y != ben)
    pick = att[stratified_sample(te.y[att], min(args.n, len(att)), rng, 200)]
    X0 = np.asarray(te.X[pick]).astype(np.float32)
    y0 = te.y[pick]

    to_raw = lambda Z: (lambda u: np.sign(u) * np.expm1(np.abs(u)))(Z * sc["std"] + sc["mean"])
    to_z = lambda R: (np.sign(R) * np.log1p(np.abs(R)) - sc["mean"]) / sc["std"]
    R0 = to_raw(X0.astype(np.float64))

    def perturb(s, t):
        R = R0.copy()
        for f in SIZE:
            R[:, ix[f]] *= s
        R[:, ix["Magnitue"]] *= np.sqrt(s)
        R[:, ix["Covariance"]] *= s * s
        for f in TIME_MUL:
            R[:, ix[f]] *= t
        for f in TIME_DIV:
            R[:, ix[f]] /= t
        return to_z(R).astype(np.float32)

    # models (all trained under the strict protocol; int8 calibration chosen on validation as in 13_int8.py)
    _s = importlib.util.spec_from_file_location("i8", os.path.join(os.path.dirname(__file__), "13_int8.py"))
    i8 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i8)
    models = {}
    st, S = load_student(run_dir(P, ds, task, "shield", 0), dev)
    st = st.cpu()
    cols = lambda X: X if len(S) == X.shape[1] else X[:, S]
    models["shield_fp32"] = lambda X: predict_logits(st, cols(X), i8.CPU)
    f1 = lambda y, z: classification_metrics(y, z, d.n_classes)["macro_f1"]
    cs = {"prop": cols(np.asarray(tr.X[stratified_sample(tr.y, 200_000, rng, 50)])),
          "bal": cols(np.asarray(tr.X[i8.balanced_sample(tr.y, 200_000 // d.n_classes, rng)]))}
    vsub = stratified_sample(va.y, 400_000, rng, 50)
    q8, _, _ = i8.choose_calibration(st, cs, cols(np.asarray(va.X[vsub])), va.y[vsub], 8, f1)
    models["shield_int8"] = lambda X: q8.forward(cols(X))
    tnet, _ = load_teacher(run_dir(P, ds, task, "teacher", 0), dev)
    models["teacher_resmlp"] = lambda X: predict_logits(tnet, X, dev, 65536, amp=True)
    xr = run_dir(P, ds, task, "xgboost", 0)
    xb = load_xgb(xr)                                       # load the ~0.8 GB booster once, not per call
    models["xgboost"] = lambda X: xgb_logits(xr, X, d.n_classes, loaded=xb)

    res = {}
    for name, fn in models.items():
        log.info("%s: scoring %d perturbation grids", name, len(S_GRID) * len(T_GRID))
        base = fn(X0).argmax(1)
        det = base != ben                                   # detected as attack when unperturbed
        evade = {}                                          # (s, t) -> mask of flows labelled Benign
        for s in S_GRID:
            for t in T_GRID:
                evade[(s, t)] = fn(perturb(s, t)).argmax(1) == ben
        rows = {}
        for Smax in (1.0, 1.25, 2.0, 10.0):
            for Tmax in (1.0, 2.0, 10.0, 30.0):
                m = np.zeros(len(y0), bool)
                for (s, t), e in evade.items():
                    if s <= Smax and 1 / Tmax <= t <= Tmax:
                        m |= e
                ok = det & m
                fam = {}
                for c in np.unique(y0):
                    k = det & (y0 == c)
                    if k.sum():
                        fam[d.classes[c]] = float(ok[k].sum() / k.sum())
                rows[f"S{Smax}_T{Tmax}"] = {"evasion_rate": float(ok.sum() / max(det.sum(), 1)), "per_class": fam}
        res[name] = {"detected_unperturbed": float(det.mean()), "budgets": rows}
        log.info("%s: detected %.3f | evasion %s", name, det.mean(),
                 {k: round(100 * v["evasion_rate"], 1) for k, v in rows.items()})
        save_json({"grid_s": S_GRID, "grid_t": T_GRID, "n_flows": int(len(y0)), "models": res},
                  P.v3 / "adversarial" / f"{ds}_{task}.json")


if __name__ == "__main__":
    main()
