"""Is the integer-only student more fragile than fp32 under input corruption? (A claim made in the literature
for quantised IDS models; tested here, no leakage.) For each student seed: fp32, W8A8 PTQ and W8A16 PTQ, all
calibrations chosen on VALIDATION exactly as in 13_int8.py (PTQ only, so the models are reproducible). The
TEST inputs are corrupted with identical, seeded noise for every model:
  gauss_s   x + N(0, (s * sd_j)^2), sd_j = per-feature train std
  impulse_p a fraction p of entries replaced by +-3 sd_j around the train mean
  drop_p    a fraction p of entries set to the train mean (missing / zeroed measurements)
Reports macro-F1 per model and condition; the drop relative to clean is compared between fp32 and int8 with a
paired test over seeds."""

import importlib.util

import numpy as np

from shield.cli import parser, registry, selected_datasets
from shield.data.datasets import load_processed, predict_logits, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.eval.predictions import load_student, run_dir
from shield.models.quant import IntMLP, calibrate
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("noise")
_s = importlib.util.spec_from_file_location("i8", __file__.replace("30_noise_robustness", "13_int8"))
i8 = importlib.util.module_from_spec(_s); _s.loader.exec_module(i8)
CONDS = [("clean", 0)] + [("gauss", s) for s in (0.05, 0.1, 0.2, 0.5)] + [("impulse", p) for p in (0.01, 0.05)] \
        + [("drop", p) for p in (0.05, 0.2)]


def corrupt(X, kind, a, mu, sd, rng):
    if kind == "clean":
        return X
    X = X.copy()
    if kind == "gauss":
        return (X + rng.normal(0, 1, X.shape).astype(np.float32) * (a * sd)).astype(np.float32)
    m = rng.random(X.shape) < a
    if kind == "impulse":
        val = mu + 3 * sd * np.where(rng.random(X.shape) < 0.5, -1, 1)
        X[m] = val[m]
    else:
        X[m] = np.broadcast_to(mu, X.shape)[m]
    return X


def main():
    p = parser(__doc__)
    p.add_argument("--variant", default="shield")
    args = p.parse_args()
    paths, reg, device = Paths(args.smoke), registry(args), get_device(args.cpu)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    cpu = i8.CPU
    for ds in [d for d in selected_datasets(args) if d in reg.resolve("main")]:
        for task in reg.tasks(ds, args.task):
            out = paths.v3 / "noise" / f"{ds}_{task}_{args.variant}_clipctl.json"
            if out.exists() and not args.force:
                continue
            data = load_processed(paths.processed(ds), task)
            tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
            f1 = lambda y, z: classification_metrics(y, z, data.n_classes)["macro_f1"]
            res = {}
            for seed in reg.seeds(ds, kcfg["seeds"]):
                run = run_dir(paths, ds, task, args.variant, seed, tcfg["model"])
                if not (run / "student.pt").exists():
                    continue
                model, S = load_student(run, device)
                model = model.cpu()
                cols = lambda X: X if len(S) == X.shape[1] else np.asarray(X[:, S])
                rng = np.random.default_rng(seed)
                Xs = cols(np.asarray(tr.X[stratified_sample(tr.y, 200_000, rng, 50)]))
                cal_sets = {"prop": Xs, "bal": cols(np.asarray(tr.X[i8.balanced_sample(tr.y, 200_000 // data.n_classes, rng)]))}
                mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
                Xv, Xt = cols(va.X), cols(te.X)
                q8, c8, _ = i8.choose_calibration(model, cal_sets, Xv, va.y, 8, f1)
                q16, c16, _ = i8.choose_calibration(model, cal_sets, Xv, va.y, 16, f1)
                lim = np.asarray(q8.s_in, np.float32) * q8.in_lim     # the W8A8 input clipping range
                models = {"fp32": lambda X: predict_logits(model, X, cpu), "w8a8": q8.forward, "w8a16": q16.forward,
                          "fp32_clip": lambda X: predict_logits(model, np.clip(X, -lim, lim).astype(np.float32), cpu)}
                r = {}
                for kind, a in CONDS:
                    Xn = corrupt(Xt, kind, a, mu, sd, np.random.default_rng(1000 + seed))
                    r[f"{kind}_{a}"] = {m: f1(te.y, fn(Xn)) for m, fn in models.items()}
                res[seed] = {"calibration": {"w8a8": c8, "w8a16": c16}, "f1": r}
                log.info("%s_%s s%d: %s", ds, task, seed,
                         {k: {m: round(100 * v, 2) for m, v in d.items()} for k, d in r.items()})
            # paired comparison of the corruption-induced drop (clean - corrupted), int8 vs fp32, over seeds
            summ = {}
            for k in [f"{kind}_{a}" for kind, a in CONDS[1:]]:
                d = {m: np.array([100 * (res[s]["f1"]["clean_0"][m] - res[s]["f1"][k][m]) for s in res])
                     for m in ("fp32", "w8a8", "w8a16", "fp32_clip")}
                summ[k] = {f"drop_{m}": float(v.mean()) for m, v in d.items()}
                summ[k]["extra_drop_w8a8_mean"] = float((d["w8a8"] - d["fp32"]).mean())
                summ[k]["extra_drop_w8a8_sd"] = float((d["w8a8"] - d["fp32"]).std(ddof=1)) if len(res) > 1 else None
                summ[k]["extra_drop_w8a16_mean"] = float((d["w8a16"] - d["fp32"]).mean())
                summ[k]["extra_drop_fp32clip_mean"] = float((d["fp32_clip"] - d["fp32"]).mean())
                summ[k]["extra_drop_w8a8_vs_clip_mean"] = float((d["w8a8"] - d["fp32_clip"]).mean())
            save_json({"seeds": res, "summary": summ}, out)
            log.info("%s_%s summary (extra drop of int8 over fp32, pts): %s", ds, task,
                     {k: (round(v["extra_drop_w8a8_mean"], 2), round(v["extra_drop_fp32clip_mean"], 2), round(v["extra_drop_w8a8_vs_clip_mean"], 2)) for k, v in summ.items()})


if __name__ == "__main__":
    main()
