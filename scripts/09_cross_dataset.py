"""Cross-dataset generalisation (configs/cross.yaml) -> outputs/cross/.

Needs the shared datasets (xciciot, xciciomt) trained through stages 03-05, and the main CIC datasets
processed (for the never-seen attack families)."""

import pandas as pd

from shield.cli import paths_for, parser, registry
from shield.cross.transfer import run_block, shap_agreement
from shield.utils.config import load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("cross")


def summarise(out_dir) -> None:
    zs, fs, un = [], [], []
    for f in sorted(out_dir.glob("*_to_*_s*.json")):
        r = load_json(f)
        key = {"source": r["source"], "target": r["target"], "task": r["task"], "seed": r["seed"]}
        zs += [{**key, **{k: v for k, v in z.items() if k != "recall"}} for z in r["zero_shot"]]
        fs += [{**key, **{k: v for k, v in z.items() if k != "recall"}} for z in r["fewshot"]]
        un += [{**key, **u} for u in r["unseen"]]
    if zs:
        z = pd.DataFrame(zs)
        z["drop"] = z["in_domain_macro_f1"] - z["macro_f1"]
        z.to_csv(out_dir / "zero_shot_runs.csv", index=False)
        (z.groupby(["source", "target", "task", "model", "norm"])
          [["in_domain_macro_f1", "macro_f1", "drop", "oracle_macro_f1"]].agg(["mean", "std"])
          .to_csv(out_dir / "zero_shot_summary.csv"))
    if fs:
        f = pd.DataFrame(fs)
        f.to_csv(out_dir / "fewshot_runs.csv", index=False)
        (f.groupby(["source", "target", "task", "model", "kind", "fraction"])["macro_f1"]
          .agg(["mean", "std", "count"]).to_csv(out_dir / "fewshot_summary.csv"))
    if un:
        u = pd.DataFrame(un)
        u.to_csv(out_dir / "unseen_runs.csv", index=False)
        (u.groupby(["source", "target", "model", "norm", "family"])["detection_rate"]
          .agg(["mean", "std"]).to_csv(out_dir / "unseen_summary.csv"))


def main():
    args = parser(__doc__, datasets=False).parse_args()
    paths, reg = paths_for(args), registry(args)
    cfg, tcfg = load_config("cross", args.smoke), load_config("teacher", args.smoke)
    device = get_device(args.cpu, args.smoke)
    out_dir = paths.out("cross")
    for src, tgt in cfg["pairs"]:
        for task in cfg["tasks"]:
            for seed in reg.seeds(src, tcfg["seeds"]):
                run_block(paths, cfg, src, tgt, task, seed, tcfg["model"], device, args.force)
    save_json(shap_agreement(paths, cfg["pairs"][:1], cfg["tasks"], tcfg["model"]), out_dir / "shap_agreement.json")
    summarise(out_dir)
    log.info("Cross-dataset results in %s", out_dir)


if __name__ == "__main__":
    main()
