"""P2 revision: 5-fold cross-validation of the placement methods' settings over instances (published FFCCP model).

Why: the reviewer asked for 5-fold cross-validation of the placement method. Placement search has no training data;
what can over-fit is the choice of algorithm settings. So each method gets a fixed candidate set of settings
(default, the v3 tuned settings, two variants of the tuned settings). The instances are split into 5 folds
(43 proved Killi & Rao instances, stratified by network and K, plus the 12 large Cogentco/Kdl settings spread over
the folds). For each fold, every method picks the candidate with the best mean score on the other 4 folds and is
scored once on the held-out fold. Score per run: z / reference (reference = proven optimum on the small instances,
best feasible value found by any run on the large ones); infeasible runs score 2.0.

Methods: hybrid_eho_aco, gwo_sa, pso_sa, sa, ga. 5 seeds x 20,000 evaluations per (instance, method, candidate).
Outputs: <v3>/placement_cv/runs.jsonl (resumable), cv_summary.json, cv_folds.csv."""

import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shield.cli import parser
from shield.placement.runner import run_algorithm, tuned_cfg, with_params
from shield.utils.config import Paths
from shield.utils.io import append_jsonl, read_jsonl, save_json
from shield.utils.logging import get_logger

log = get_logger("placement_cv")
_s = importlib.util.spec_from_file_location("ext", Path(__file__).with_name("58_ffccp_extended.py"))
ext = importlib.util.module_from_spec(_s); _s.loader.exec_module(ext)
kr = ext.kr

SEEDS, FOLDS, INFEASIBLE = 5, 5, 2.0
# (method, candidate) -> overrides on top of the base config; "default" = no tuned settings.
CANDIDATES = {
    "hybrid_eho_aco": {"default": None, "tuned": {}, "clans5": {"hybrid.n_clans": 5},
                       "ls_every5": {"hybrid.local_search_every": 5, "hybrid.local_search_evals": 200}},
    "gwo_sa": {"default": None, "tuned": {}, "pack30": {"gwo.pack": 30},
               "sa_every5": {"gwo_sa.every": 5, "gwo_sa.evals": 200}},
    "pso_sa": {"default": None, "tuned": {}, "swarm30": {"pso.swarm": 30}, "w07": {"pso.w": 0.7}},
    "sa": {"default": None, "tuned": {}, "cool9999": {"sa.cooling": 0.9999}, "cool999": {"sa.cooling": 0.999}},
    "ga": {"default": None, "tuned": {}, "pop20": {"ga.pop": 20}, "mut02": {"ga.mut_rate": 0.2}},
}


def instances(P):
    small, large = [], []
    for name, pmax in kr.TOPOS.items():
        D, L = kr.instance(name, P)
        for K, pm in pmax.items():
            small += [(name, K, p) for p in range(max(K, math.ceil(K * L.sum() / kr.U)), pm + 1)]
    for name in ext.LARGE:
        D, L = kr.instance(name, P)
        pmin = max(2, math.ceil(2 * L.sum() / kr.U))
        large += [(name, 2, p) for p in range(pmin, pmin + 6)]
    return small, large


def cfg_for(alg, cand):
    base = kr._W["pcfg"]
    tuned = kr._W["tuned"]
    over = CANDIDATES[alg][cand]
    if alg == "pso_sa":
        c = base if over is None else tuned_cfg(tuned_cfg(base, "pso", tuned), "gwo_sa", tuned)
    else:
        c = base if over is None else tuned_cfg(base, alg, tuned)
    return with_params(c, over or {})


def _run(task):
    name, K, p, alg, cand, seed = task
    D, L = kr.instance(name, kr._W["paths"])
    prob = kr.FFCCP(D, L, p, K)
    cfg = cfg_for(alg, cand)
    if alg == "pso_sa":
        _, F, sec = ext.run_pso_sa(prob, cfg, seed, kr.BUDGET)
    else:
        res = run_algorithm(alg, prob, kr._Topo(), cfg, seed, budget=kr.BUDGET)
        F = res.best_F
    return {"topology": name, "K": K, "p": p, "alg": alg, "cand": cand, "seed": seed, "z": float(F)}


def _init():
    ext._init()


def fold_of(small, large):
    fold = {}
    for key in sorted({(t, K) for t, K, _ in small}):
        ps = sorted(p for t, K, p in small if (t, K) == key)
        for i, p in enumerate(ps):
            fold[(key[0], key[1], p)] = i % FOLDS
    for i, inst in enumerate(sorted(large, key=lambda x: (x[2], x[0]))):
        fold[inst] = i % FOLDS
    return fold


def main():
    a = parser(__doc__, datasets=False)
    a.add_argument("--workers", type=int, default=14)
    args = a.parse_args()
    P = Paths(False)
    od = P.v3 / "placement_cv"; od.mkdir(parents=True, exist_ok=True)
    small, large = instances(P)
    runs_f = od / "runs.jsonl"
    done = {(r["topology"], r["K"], r["p"], r["alg"], r["cand"], r["seed"]) for r in read_jsonl(runs_f)}
    tasks = [(t, K, p, alg, cand, s) for t, K, p in small + large for alg in CANDIDATES
             for cand in CANDIDATES[alg] for s in range(SEEDS)]
    # small instances first so a partial run is already usable
    tasks = [x for x in tasks if x not in done]
    log.info("%d small + %d large instances, %d runs to do", len(small), len(large), len(tasks))
    if tasks:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
            for row in pool.imap_unordered(_run, tasks, chunksize=2):
                append_jsonl(row, runs_f)

    df = pd.DataFrame(read_jsonl(runs_f))
    opt = json.loads((P.v3 / "killi_rao" / "optimum.json").read_text())
    old_large = pd.DataFrame(read_jsonl(P.v3 / "killi_rao" / "large_runs.jsonl"))
    ref = {}
    for t, K, p in small:
        o = opt.get(f"{t}|{K}|{p}")
        if o and o.get("status") == 0:
            ref[(t, K, p)] = o["z"]
    for t, K, p in large:
        zs = pd.concat([df[(df.topology == t) & (df.p == p)].z, old_large[(old_large.topology == t) & (old_large.p == p)].z])
        zs = zs[zs < 1e3]
        if len(zs):
            ref[(t, K, p)] = zs.min()
    df["inst"] = list(zip(df.topology, df.K, df.p))
    df = df[df.inst.isin(ref)].copy()
    df["feasible"] = df.z < 1e3
    df["ratio"] = np.where(df.feasible, df.z / df.inst.map(ref), INFEASIBLE)
    df["gap"] = np.where(df.feasible, 100 * (df.ratio - 1), np.nan)
    df["small"] = df.topology.isin(list(kr.TOPOS))
    df["hit"] = df.feasible & (df.ratio <= 1 + 1e-9)
    fold = fold_of([i for i in small if i in ref], [i for i in large if i in ref])
    df["fold"] = df.inst.map(fold)

    per_inst = df.groupby(["alg", "cand", "inst", "fold", "small"]).agg(
        ratio=("ratio", "mean"), gap=("gap", "mean"), hit=("hit", "mean"), feasible=("feasible", "mean")).reset_index()
    rows, held = [], []
    for f in range(FOLDS):
        for alg in CANDIDATES:
            tr = per_inst[(per_inst.alg == alg) & (per_inst.fold != f)]
            choice = tr.groupby("cand").ratio.mean().idxmin()
            te = per_inst[(per_inst.alg == alg) & (per_inst.fold == f) & (per_inst.cand == choice)]
            held.append(te.assign(test_fold=f))
            s, l = te[te.small], te[~te.small]
            rows.append({"fold": f, "alg": alg, "chosen": choice, "n_test": len(te),
                         "ratio": te.ratio.mean(),
                         "small_gap": s.gap.mean(), "small_hit": 100 * s.hit.mean(),
                         "large_gap": l.gap.mean(), "large_feasible": 100 * l.feasible.mean()})
    folds = pd.DataFrame(rows)
    folds.to_csv(od / "cv_folds.csv", index=False)
    held = pd.concat(held)
    summ = {}
    for alg, g in folds.groupby("alg"):
        summ[alg] = {k: {"mean": float(g[k].mean()), "sd": float(g[k].std(ddof=1))}
                     for k in ("ratio", "small_gap", "small_hit", "large_gap", "large_feasible")}
        summ[alg]["chosen"] = g.chosen.tolist()
    piv = held.pivot_table(index="inst", columns="alg", values="ratio")
    tests = {}
    for other in ("gwo_sa", "pso_sa", "sa", "ga"):
        d = (piv["hybrid_eho_aco"] - piv[other]).dropna()
        nz = d[d.abs() > 1e-12]
        tests[other] = {"better": int((nz < 0).sum()), "worse": int((nz > 0).sum()), "ties": int(len(d) - len(nz)),
                        "p": float(wilcoxon(nz).pvalue) if len(nz) >= 6 else None}
    summ["eho_aco_vs"] = tests
    save_json(summ, od / "cv_summary.json")
    log.info("held-out folds:\n%s", folds.groupby("alg")[["ratio", "small_gap", "small_hit", "large_gap",
                                                           "large_feasible"]].agg(["mean", "std"]).round(3))
    log.info("chosen settings: %s", {k: v["chosen"] for k, v in summ.items() if k != "eho_aco_vs"})
    log.info("EHO-ACO vs others (held-out instances): %s", tests)


if __name__ == "__main__":
    main()
