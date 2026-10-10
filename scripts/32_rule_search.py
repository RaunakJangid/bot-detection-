"""Metaheuristics applied to the detector (no leakage): the same evaluation budget for every optimiser
(src/shield/eval/rule_search.py), fitted on VALIDATION half A, checked on half B, test scored once per run.

  bias     per-class logit bias of the best ensemble (ens4), grid -4..4 step 0.2; x0 = no bias
  weights  ensemble member weights 0..5 per member (probability averaging), x0 = equal weights; the
           Caruana greedy selection (current method) fitted on the same half A is the reference
Seeds change the A/B split and the optimiser's random stream. Paired Wilcoxon over seeds vs eho_aco."""

import importlib.util
import json

import numpy as np
from scipy.stats import wilcoxon

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import softmax
from shield.eval.predictions import model_logits
from shield.eval.rule_search import METHODS, run
from shield.utils.config import Paths
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("rule_search")
_s = importlib.util.spec_from_file_location("dv4", __file__.replace("32_rule_search", "19_detector_v4"))
dv4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(dv4)


def macro_f1(y, pred, n):
    tp = np.bincount(y[pred == y], minlength=n); pc = np.bincount(pred, minlength=n); tc = np.bincount(y, minlength=n)
    f = 2 * tp / np.maximum(pc + tc, 1)
    return f[tc > 0].mean()


def main():
    p = parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--rows", type=int, default=400_000, help="stratified rows of half A used by the optimisers")
    p.add_argument("--problems", nargs="*", default=["bias", "weights"])
    args = p.parse_args()
    paths, device = Paths(args.smoke), get_device(args.cpu)
    ds, task = args.dataset, args.task
    data = load_processed(paths.processed(ds), task)
    n, yv, yt = data.n_classes, data.splits["validation"].y, data.splits["test"].y
    SPL = ("validation", "test")
    lc = paths.v3 / "logits" / f"{ds}_{task}"
    members = {"teacher": [model_logits(paths, data, "teacher", s, SPL, device) for s in range(5)],
               "xgboost": [model_logits(paths, data, "xgboost", 0, SPL, device)]}
    for m in ("lgbm", "catboost", "tabm"):
        if dv4.have(paths, ds, task, f"{m}_s0"):
            members[m] = [dv4.load(paths, ds, task, f"{m}_s0")]
    for m in ("qt_ce", "qt_bs"):
        members[m] = [dv4.load(paths, ds, task, f"{m}_s{s}") for s in range(3)]
    Pm = {m: {s: np.mean([softmax(z[s]) for z in zs], 0).astype(np.float32) for s in SPL} for m, zs in members.items()}
    names = list(Pm)
    ens = {s: np.load(lc / f"ens4_s0_{s}.npy").astype(np.float32) for s in SPL}
    grid = np.round(np.arange(-4, 4.0001, 0.2), 2)
    out = {"members": names, "grid": grid.tolist(), "runs": []}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        A = np.zeros(len(yv), bool); A[stratified_sample(yv, len(yv) // 2, rng, 1)] = True
        iA = np.flatnonzero(A); iB = np.flatnonzero(~A)
        sub = iA[stratified_sample(yv[iA], min(args.rows, len(iA)), rng, 2000)]   # rare classes kept in full
        for prob in args.problems:
            if prob == "bias":
                Z = {"A": ens["validation"][sub], "B": ens["validation"][iB], "T": ens["test"]}
                f = lambda x, k="A", y=yv[sub]: macro_f1(y, (Z[k] + grid[x]).argmax(1), n)
                D, G, x0 = n, len(grid), np.full(n, int(np.argmin(np.abs(grid))))
                eta = np.exp(-np.abs(grid))                            # ant heuristic: prefer small biases
            else:
                S = {"A": np.stack([Pm[m]["validation"][sub] for m in names]),
                     "B": np.stack([Pm[m]["validation"][iB] for m in names]),
                     "T": np.stack([Pm[m]["test"] for m in names])}
                f = lambda x, k="A", y=yv[sub]: macro_f1(y, np.tensordot(x.astype(np.float32), S[k], 1).argmax(1), n) \
                    if x.sum() > 0 else 0.0
                D, G, x0 = len(names), 6, np.full(len(names), 1)
                eta = np.ones(G)
            for meth in METHODS:
                kw = {"eta": eta} if meth == "eho_aco" else {}
                obj = run(meth, f, D, G, x0, args.budget, seed, **kw)
                x = obj.best_x
                rec = {"seed": seed, "problem": prob, "method": meth, "valA": obj.best_f, "evals": obj.n,
                       "valB": f(x, "B", yv[iB]), "test": f(x, "T", yt), "x": x.tolist(),
                       "trace": obj.trace[:: max(1, len(obj.trace) // 100)]}
                out["runs"].append(rec)
                log.info("s%d %s %-8s valA %.4f valB %.4f test %.4f", seed, prob, meth, rec["valA"], rec["valB"], rec["test"])
            if prob == "weights":   # Caruana greedy (the current ens4 method) on the same rows, for reference
                counts, _ = dv4.ensemble_selection({m: Pm[m]["validation"][sub] for m in names}, yv[sub], n)
                x = np.array([counts.get(m, 0) for m in names])
                out["runs"].append({"seed": seed, "problem": prob, "method": "caruana", "valA": f(x),
                                    "valB": f(x, "B", yv[iB]), "test": f(x, "T", yt), "x": x.tolist()})
            save_json(out, paths.v3 / "rule_search" / f"{ds}_{task}.json")
    summ = {}
    for prob in args.problems:
        R = [r for r in out["runs"] if r["problem"] == prob]
        ms = sorted({r["method"] for r in R})
        for m in ms:
            v = {k: np.array([r[k] for r in R if r["method"] == m]) for k in ("valB", "test")}
            summ[f"{prob}/{m}"] = {k: [float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0] for k, a in v.items()}
        e = np.array([r["valB"] for r in R if r["method"] == "eho_aco"])
        for m in ms:
            o = np.array([r["valB"] for r in R if r["method"] == m])
            if m != "eho_aco" and len(o) == len(e) and len(e) >= 5 and np.any(e != o):
                summ[f"{prob}/{m}"]["wilcoxon_p_vs_eho_aco_valB"] = float(wilcoxon(e, o).pvalue)
    out["summary"] = summ
    save_json(out, paths.v3 / "rule_search" / f"{ds}_{task}.json")
    log.info("summary %s", json.dumps({k: {kk: [round(100 * x, 2) for x in vv] if isinstance(vv, list) else vv
                                           for kk, vv in v.items()} for k, v in summ.items()}))


if __name__ == "__main__":
    main()
