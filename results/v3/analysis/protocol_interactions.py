"""Main effects and interactions of the E4 protocol factorial (scripts/39_protocol_ablation.py output).

Effects in macro-F1 / accuracy points, per task and model. Main effect of a factor = mean over the other factors'
levels of (leaky level - strict level). Two-way interaction = half the difference of the factor's effect between the
levels of the other factor (standard 2^k contrast). CICIoT2023 has no split factor (no capture identifiers)."""
import itertools
import json
import sys

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else r"D:\shield_run\outputs_v3\ablation\protocol_ablation.json"))
LEVELS = {"split": ("official", "random"), "IAT": ("removed", "kept"), "dedup": ("on", "off")}   # (strict, leaky)
out = {}
for task in ("ciciot_class34", "ciciomt_category", "ciciomt_attack"):
    for model in ("xgboost", "rf"):
        factors = ["IAT", "dedup"] + (["split"] if task.startswith("ciciomt") else [])
        cell = {}
        for combo in itertools.product(*[(0, 1)] * len(factors)):
            lv = {f: LEVELS[f][c] for f, c in zip(factors, combo)}
            split = lv.get("split", "official")
            k = f"{task}|split={split}|IAT={lv['IAT']}|dedup={lv['dedup']}|{model}|s0"
            cell[combo] = R[k]
        res = {}
        for metric in ("macro_f1", "accuracy"):
            v = {c: 100 * r[metric] for c, r in cell.items()}
            n = len(factors)
            eff = {}
            for i, f in enumerate(factors):
                diffs = [v[c[:i] + (1,) + c[i + 1:]] - v[c[:i] + (0,) + c[i + 1:]] for c in v if c[i] == 0]
                eff[f] = sum(diffs) / len(diffs)
            for i, j in itertools.combinations(range(n), 2):
                def e(fix):
                    ds = [v[c[:i] + (1,) + c[i + 1:]] - v[c[:i] + (0,) + c[i + 1:]] for c in v if c[i] == 0 and c[j] == fix]
                    return sum(ds) / len(ds)
                eff[f"{factors[i]}x{factors[j]}"] = (e(1) - e(0)) / 2
            res[metric] = {k: round(x, 2) for k, x in eff.items()}
            res[metric]["strict"] = round(v[(0,) * n], 2)
            res[metric]["all_leaky"] = round(v[(1,) * n], 2)
        out[f"{task}|{model}"] = res
        print(task, model, json.dumps(res))
json.dump(out, open(r"D:\shield_run\outputs_v3\ablation\protocol_interactions.json", "w"), indent=2)
