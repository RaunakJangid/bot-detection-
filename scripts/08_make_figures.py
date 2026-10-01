"""Build every paper table and figure from the saved outputs -> outputs/paper/."""

import numpy as np
import pandas as pd

from shield.cli import parser, paths_for, selected_datasets, tasks_for
from shield.eval import plots
from shield.eval.tables import detection_table, k_sweep_table, save_table
from shield.utils.config import load_config
from shield.utils.io import load_json, read_jsonl
from shield.utils.logging import get_logger

log = get_logger("figures")


def detection(paths, out, args):
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    lat_file = paths.outputs / "latency" / "latency.json"
    lat = load_json(lat_file) if lat_file.exists() else {}
    for ds in selected_datasets(args):
        for task in tasks_for(tcfg["tasks"], ds, args.task):
            key = f"{ds}_{task}"
            df = detection_table(paths, ds, task, tcfg["model"], list(kcfg["variants"]))
            if df.empty:
                continue
            save_table(df, out / "tables" / f"detection_{key}", f"Detection results, {key}")
            if key in lat:
                plots.pareto(lat[key], out / "figures" / f"pareto_{key}", key)
            ks = k_sweep_table(paths, ds, task)
            teacher_f1 = None
            tdir = paths.outputs / "teacher" / f"{key}_{tcfg['model']}_s{kcfg['teacher_seed']}"
            if (tdir / "metrics.json").exists():
                tm = load_json(tdir / "metrics.json")
                teacher_f1 = tm["test"]["macro_f1"]
                classes = list(tm["test"]["per_class"])
                plots.confusion_heatmap(np.load(tdir / "confusion.npy"), classes,
                                        out / "figures" / f"confusion_teacher_{key}", f"Teacher, {key}")
                sdir = paths.outputs / "kd" / key / f"shield_s{kcfg['seeds'][0]}"
                if (sdir / "confusion.npy").exists():
                    plots.confusion_heatmap(np.load(sdir / "confusion.npy"), classes,
                                            out / "figures" / f"confusion_shield_{key}", f"SHIELD student, {key}")
            if not ks.empty:
                save_table(ks, out / "tables" / f"k_sweep_{key}", f"F1 vs k, {key}")
                plots.f1_vs_k(ks, out / "figures" / f"f1_vs_k_{key}", key, teacher_f1)
            log.info("detection %s: %d rows", key, len(df))


def placement(paths, out, args):
    pcfg = load_config("placement", args.smoke)
    pdir = paths.outputs / "placement"
    if not (pdir / "summary.csv").exists():
        log.warning("no placement results yet")
        return
    summary = pd.read_csv(pdir / "summary.csv")
    runs = [r for r in read_jsonl(pdir / "runs.jsonl") if not r.get("skipped")]
    df = pd.read_parquet(pdir / "runs.parquet")
    piv = summary.pivot_table(index=["topology", "k", "rho"], columns="algorithm", values="F_mean").reset_index()
    save_table(piv, out / "tables" / "placement_objective", "Mean objective F per instance")
    if "gap_pct" in summary:
        gaps = summary.dropna(subset=["gap_pct"]).pivot_table(index=["topology", "k", "rho"], columns="algorithm",
                                                              values="gap_pct").reset_index()
        if not gaps.empty:
            save_table(gaps, out / "tables" / "placement_gap", "Optimality gap (%) vs brute force")
    if (pdir / "wilcoxon.csv").exists():
        w = pd.read_csv(pdir / "wilcoxon.csv")
        if not w.empty:
            agg = w.groupby("vs").agg(instances=("p", "size"), significant=("p_holm", lambda p: int((p < 0.05).sum())),
                                      wins=("wins", "sum"), ties=("ties", "sum"), losses=("losses", "sum")).reset_index()
            save_table(agg, out / "tables" / "placement_wilcoxon", "Hybrid EHO-ACO vs baselines (Wilcoxon, Holm)")
    rho = pcfg["rho_default"]
    plots.objective_vs_k(summary, rho, out / "figures")
    k_mid = pcfg["rho_sweep_k"] or pcfg["k_values"][len(pcfg["k_values"]) // 2]
    for topo in sorted(df.topology.unique()):
        plots.convergence(runs, topo, k_mid, rho, pcfg["budget"], out / "figures" / f"convergence_{topo}")
    for metric, label in (("res_controller_single_worst_avg_ms", "worst single-failure avg latency (ms)"),
                          ("res_link_random_10_mean_pct_sla", "switches within SLA, 10% random link failures")):
        plots.resilience_bars(df, k_mid, rho, metric, out / "figures" / f"resilience_{metric}", label)
    cols = [c for c in df.columns if c.startswith("res_")]
    res = df[df.k == k_mid].groupby(["topology", "algorithm"])[cols].mean().reset_index()
    save_table(res, out / "tables" / "placement_resilience", f"Resilience metrics, k={k_mid}")


def coupled(paths, out, args):
    cdir = paths.outputs / "coupled"
    if not (cdir / "min_controllers.csv").exists():
        return
    mink = pd.read_csv(cdir / "min_controllers.csv").set_index(["topology", "rho"])
    save_table(mink.reset_index(), out / "tables" / "coupled_min_controllers",
               "Controllers needed to meet the SLA, per detector")
    plots.coupled_bars(mink, out / "figures" / "coupled_min_controllers", load_json(cdir / "coupled_meta.json")["k_max"])


def main():
    args = parser(__doc__).parse_args()
    paths = paths_for(args)
    out = paths.out("paper")
    detection(paths, out, args)
    placement(paths, out, args)
    coupled(paths, out, args)
    log.info("Paper tables and figures in %s", out)


if __name__ == "__main__":
    main()
