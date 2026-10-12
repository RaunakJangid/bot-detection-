"""E2: what drives false-alarm inflation after a change of capture: model family, architecture, distillation or seed?

Input: outputs_v3/shift/shift_base_rate.json and outputs_v3/recalib/recalib.json (no new training).
Response: log10 test FPR at the 1% (and 0.1%) validation target, per model run.
  * Seed vs architecture: for architectures with >= 3 seeds, one-way random-effects ANOVA; ICC(1) = share of
    variance between architectures; the rest is seed-to-seed variance within an architecture.
  * Family: Kruskal-Wallis over families (gbdt, ensemble, large neural, small MLP scratch, small MLP KD) using one
    value per architecture (mean over its seeds), so seeds are not counted as independent.
  * Distillation: size-matched small MLPs, KD vs scratch, seed-level Mann-Whitney (reported, not a headline).
  * The same ICC after label-free recalibration (corrected, K=2000, pi=1%) shows whether recalibration removes
    the seed lottery."""

import re
from collections import defaultdict

import numpy as np
from scipy.stats import f as fdist, kruskal, mannwhitneyu

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("fpr_var")
TASKS = ("ciciomt_binary", "ciciomt_category", "ciciomt_attack", "ciciot_class34")


def family(b):
    if b in ("xgboost", "xgbsub", "catboost", "lgbm", "lgbm_tuned", "two_stage"):
        return "gbdt"
    if b in ("ensemble", "ens4", "treeens"):
        return "ensemble"
    if b in ("scratch_all", "mlp_s"):
        return "small_mlp_scratch"
    if b.startswith(("kd_", "shield", "v3-")):
        return "small_mlp_kd"
    return "large_neural"


def icc1(groups):
    """One-way random-effects ICC(1) with unequal group sizes; returns icc, F, p."""
    g = [np.asarray(x) for x in groups if len(x) >= 2]
    k, N = len(g), sum(len(x) for x in g)
    grand = np.concatenate(g).mean()
    ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in g)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(len(x) ** 2 for x in g) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    F = msb / msw
    return float(icc), float(F), float(fdist.sf(F, k - 1, N - k))


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    S = load_json(P.v3 / "shift" / "shift_base_rate.json")
    R = load_json(P.v3 / "recalib" / "recalib.json")
    out = {}
    for t in TASKS:
        res = {}
        for name, get in (("fixed_1pct", lambda m, v: v["fpr@0.01"]["test_fpr"]),
                          ("fixed_0.1pct", lambda m, v: v["fpr@0.001"]["test_fpr"]),
                          ("corrected_1pct", lambda m, v: R[t][m]["corrected_K2000_pi0.01"]["fpr_median"])):
            arch = defaultdict(list)
            for m, v in S[t]["models"].items():
                arch[re.sub(r"_s\d+$", "", m)].append(np.log10(max(get(m, v), 1e-4)))
            multi = {a: x for a, x in arch.items() if len(x) >= 3}
            icc, F, p = icc1(list(multi.values()))
            fam = defaultdict(list)
            for a, x in arch.items():
                fam[family(a)].append(float(np.mean(x)))
            kw = kruskal(*[x for x in fam.values() if len(x) >= 2])
            within_sd = float(np.sqrt(np.mean([np.var(x, ddof=1) for x in multi.values()])))
            between_sd = float(np.std([np.mean(x) for x in multi.values()], ddof=1))
            kd = [y for a, x in arch.items() if family(a) == "small_mlp_kd" for y in x]
            sc = [y for a, x in arch.items() if a == "scratch_all" for y in x]
            res[name] = {"icc_between_architectures": icc, "anova_F": F, "anova_p": p,
                         "n_architectures_multiseed": len(multi), "within_arch_sd_log10": within_sd,
                         "between_arch_sd_log10": between_sd,
                         "kruskal_family_p": float(kw.pvalue),
                         "family_median_fpr_pct": {f: float(100 * 10 ** np.median(x)) for f, x in fam.items()},
                         "kd_vs_scratch_p": float(mannwhitneyu(kd, sc).pvalue) if len(sc) >= 3 else None,
                         "per_arch_range_pct": {a: [float(100 * 10 ** min(x)), float(100 * 10 ** max(x))]
                                                for a, x in multi.items()}}
            log.info("%s %s: ICC %.2f (F=%.2f p=%.2g; %d archs) | within-arch sd %.2f vs between sd %.2f (log10) | "
                     "family KW p=%.2g | KD vs scratch p=%s", t, name, icc, F, p, len(multi), within_sd, between_sd,
                     kw.pvalue, None if res[name]["kd_vs_scratch_p"] is None else round(res[name]["kd_vs_scratch_p"], 3))
        out[t] = res
    save_json(out, P.v3 / "fpr_variance" / "fpr_variance.json")


if __name__ == "__main__":
    main()
