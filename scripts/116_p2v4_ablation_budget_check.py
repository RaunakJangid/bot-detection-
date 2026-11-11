"""P2 v4: evaluation-budget check for the ablations (scripts/105 FFCCP, scripts/106 shifted). The run records do not store
the number of objective evaluations, so one task per variant per benchmark is re-run here with the identical code path and
Result.n_evals (every Evaluator call counted, cache hits included; SA intensification calls go through the same Evaluator)
is reported, together with a check that the re-run reproduces the stored F of that task.
Output: outputs_v5/p2/ablation_budget_check.json."""
import importlib.util
import json
from pathlib import Path

OUT = Path("D:/shield_run/outputs_v5/p2")
_l = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(f[:-3], Path(__file__).with_name(f)))
m105, m106 = _l("105_p2v4_ehoaco_ablation.py"), _l("106_p2v4_ehoaco_ablation_shift.py")


def main():
    from shield.placement.runner import tuned_cfg
    res = {"ffccp": {}, "shift": {}}
    m105._init(); kr = m105._M["m63"].kr
    stored = {(r["topology"], r["K"], r["p"], r["alg"], r["seed"]): r["z"] for r in map(json.loads, open(OUT / "ablation_runs.jsonl"))}
    name, K, p = "Cogentco", 2, 13
    D, L = kr.instance(name, kr._W["paths"])
    for alg in m105.VARIANTS:
        prob = kr.FFCCP(D, L, p, K); cfg = tuned_cfg(kr._W["pcfg"], "hybrid_eho_aco", kr._W["tuned"])
        r = m105.run_variant(alg, prob, kr._Topo(), cfg, 0, kr.BUDGET)
        res["ffccp"][alg] = {"n_evals": int(r.n_evals), "budget": kr.BUDGET, "F": float(r.best_F), "stored_F": stored.get((name, K, p, alg, 0))}
        print("ffccp", alg, res["ffccp"][alg], flush=True)
    m106._init()
    rows = [json.loads(l) for l in open(OUT / "modelA_v4.jsonl")]
    x = next(x for x in rows if x["seed"] == 7 and any(x["k_env"].get(sc) is not None for sc in m106.m94.SCEN))
    sc = next(sc for sc in m106.m94.SCEN if x["k_env"].get(sc) is not None)
    from shield.placement.coupling import topology_sla  # noqa: F401  (imported by _run)
    for alg in m106.VARIANTS:
        t = (x["topology"], x["rho"], x["seed"], sc, int(x["k_env"][sc]), float(x["ratios"][sc]), alg, 0)
        out = m106._run(t)
        # n_evals: re-run the same problem through run_variant to read the counter
        res["shift"][alg] = {"task": list(map(str, t)), "F": out["F"]}
    import shield.placement.common as common
    orig = common.Evaluator.result
    counts = []
    common.Evaluator.result = lambda self: (counts.append(self.n_evals), orig(self))[1]
    for alg in m106.VARIANTS:
        t = (x["topology"], x["rho"], x["seed"], sc, int(x["k_env"][sc]), float(x["ratios"][sc]), alg, 0)
        counts.clear(); out = m106._run(t)
        res["shift"][alg].update({"n_evals": counts[-1] if counts else None, "budget": m106.BUDGET, "F_rerun": out["F"]})
        print("shift", alg, res["shift"][alg], flush=True)
    json.dump(res, open(OUT / "ablation_budget_check.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
