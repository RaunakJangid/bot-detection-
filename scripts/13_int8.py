"""v3 step 4: integer-only int8 students (per-feature input scales, per-channel weights, calibrated
activations), with quantisation-aware fine-tuning when post-training quantisation loses accuracy.

Every choice is made on VALIDATION macro-F1: the calibration (grid below), and PTQ vs QAT (QAT is tried
only if PTQ loses more than `--qat-threshold` points on validation, and kept only if better there).
Reported: W8A8 (full int8) and W8A16 (int8 weights, 16-bit activations). Test is scored once.
Writes outputs_v3/int8/<dataset>_<task>/<variant>_s<seed>.json and int8/summary.csv.
"""

import copy

import numpy as np
import pandas as pd
import torch

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed, predict_logits, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.predictions import load_student, run_dir
from shield.kd.cache import load_cache_dir
from shield.kd.trainer import ensemble_cache_dir
from shield.models.quant import IntMLP, calibrate, qat_finetune
from shield.models.train import teacher_dir
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("int8")
VARIANTS = ["scratch_all", "kd_all", "kd_shap", "shield"]
CPU = torch.device("cpu")
# Calibration grid, chosen per model on validation: calibration rows (training-proportional or
# class-balanced: rare classes sit in the feature tails) x input clip percentile x activation clip percentile.
IN_PCTS, ACT_PCTS = (99.99, 99.999, 100.0), (99.99, 100.0)


def balanced_sample(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx = [rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


def choose_calibration(model, cal_sets: dict, Xv: np.ndarray, yv: np.ndarray, bits: int, f1,
                       max_rows: int = 400_000):
    """(IntMLP, (set, in_pct, act_pct), validation macro-F1) of the best grid point on (a subset of) validation."""
    sub = np.sort(np.random.default_rng(0).choice(len(yv), min(len(yv), max_rows), replace=False))
    best = None
    for name, Xc in cal_sets.items():
        for ip in IN_PCTS:
            for ap in ACT_PCTS:
                q = IntMLP(model, calibrate(model, Xc, ip, ap), act_bits=bits)
                f = f1(yv[sub], q.forward(Xv[sub]))
                if best is None or f > best[2] + 1e-6:
                    best = (q, (name, ip, ap), f)
    q, key, _ = best
    return q, key, f1(yv, q.forward(Xv))


def main():
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    p.add_argument("--v3-students", action="store_true", help="quantise the v3 students (outputs_v3/kd)")
    p.add_argument("--qat-threshold", type=float, default=0.5)
    p.add_argument("--calib-rows", type=int, default=200_000)
    args = p.parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    rows = []
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main,full")]:
        for task in reg.tasks(ds, args.task):
            data = load_processed(paths.processed(ds), task)
            tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
            for v in args.variants:
                for seed in reg.seeds(ds, kcfg["seeds"]):
                    out = paths.v3 / "int8" / f"{ds}_{task}" / f"{v}_s{seed}.json"
                    if out.exists() and not args.force:
                        rows.append(load_json(out)["row"])
                        continue
                    run = (paths.v3 / "kd" / f"{ds}_{task}" / f"{v}_s{seed}" if args.v3_students
                           else run_dir(paths, ds, task, v, seed, tcfg["model"]))
                    if not (run / "student.pt").exists():
                        continue
                    model, S = load_student(run, device)
                    model = model.cpu()
                    spec = load_json(run / "metrics.json")["spec"]
                    cols = lambda X: X if len(S) == X.shape[1] else np.asarray(X[:, S])
                    rng = np.random.default_rng(seed)
                    cal_sets = {"prop": cols(np.asarray(tr.X[stratified_sample(tr.y, args.calib_rows, rng, 50)])),
                                "bal": cols(np.asarray(tr.X[balanced_sample(tr.y, args.calib_rows // data.n_classes, rng)]))}
                    Xv, Xt = cols(va.X), cols(te.X)
                    f1 = lambda y, z: classification_metrics(y, z, data.n_classes)["macro_f1"]

                    fp_v = f1(va.y, predict_logits(model, Xv, CPU))
                    fp_t = classification_metrics(te.y, predict_logits(model, Xt, CPU),
                                                  data.n_classes, data.classes, full=True)
                    ptq, cal_name, ptq_v = choose_calibration(model, cal_sets, Xv, va.y, 8, f1)
                    chosen, q_v, mode = ptq, ptq_v, "ptq"
                    if 100 * (fp_v - ptq_v) > args.qat_threshold:
                        qm = copy.deepcopy(model)
                        t_logits = None
                        if spec.get("beta", 0) > 0:
                            cache = (ensemble_cache_dir(paths, ds, task) if spec.get("teacher") == "ensemble"
                                     else teacher_dir(paths, ds, task, tcfg["model"], kcfg["teacher_seed"]) / "cache")
                            t_logits = load_cache_dir(cache)[0]
                        cset, ip, ap = cal_name
                        qat_finetune(qm.to(device), calibrate(qm, cal_sets[cset], ip, ap), tr.X, tr.y, t_logits,
                                     device, seed=seed, T=spec.get("T", 1.0),
                                     alpha=spec.get("alpha", 0.3), beta=spec.get("beta", 0.7), cols=S)
                        qm = qm.cpu()
                        qat, qat_cal, qat_v = choose_calibration(qm, cal_sets, Xv, va.y, 8, f1)
                        if qat_v > ptq_v:
                            chosen, q_v, mode, cal_name = qat, qat_v, "qat", qat_cal
                    q_t = classification_metrics(te.y, chosen.forward(Xt), data.n_classes, data.classes, full=True)
                    w8a16, cal16, w16_v = choose_calibration(model, cal_sets, Xv, va.y, 16, f1)
                    w16_t = f1(te.y, w8a16.forward(Xt))
                    v2 = load_json(run / "metrics.json").get("test_int8", {}).get("macro_f1")
                    row = {"dataset": ds, "task": task, "variant": v, "seed": seed, "mode": mode,
                           "calibration_w8a8": "/".join(map(str, cal_name)), "calibration_w8a16": "/".join(map(str, cal16)),
                           "val_fp32": fp_v, "val_ptq": ptq_v, "val_int8": q_v,
                           "test_fp32": fp_t["macro_f1"], "test_int8": q_t["macro_f1"], "test_int8_v2_dynamic": v2,
                           "drop_pts": 100 * (fp_t["macro_f1"] - q_t["macro_f1"]),
                           "drop_pts_v2": None if v2 is None else 100 * (fp_t["macro_f1"] - v2),
                           "val_w8a16": w16_v, "test_w8a16": w16_t, "drop_pts_w8a16": 100 * (fp_t["macro_f1"] - w16_t),
                           "int8_bytes": chosen.size_bytes(), "fp32_bytes": 4 * sum(p.numel() for p in model.parameters())}
                    save_json({"row": row, "test_int8": q_t}, out)
                    log.info("%s_%s %s s%d: fp32 %.4f | W8A8 %s %.4f (drop %.2f) | W8A16 drop %.2f | v2 drop %s | %d B",
                             ds, task, v, seed, row["test_fp32"], mode, row["test_int8"], row["drop_pts"],
                             row["drop_pts_w8a16"],
                             "n/a" if v2 is None else f"{row['drop_pts_v2']:.2f}", row["int8_bytes"])
                    rows.append(row)
    df = pd.DataFrame(rows)
    paths.v3.joinpath("int8").mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.v3 / "int8" / ("summary_v3students.csv" if args.v3_students else "summary.csv"), index=False)
    if len(df):
        log.info("\n%s", df.groupby(["dataset", "task"])[["drop_pts", "drop_pts_w8a16", "drop_pts_v2"]]
                 .mean().round(2).to_string())


if __name__ == "__main__":
    main()
