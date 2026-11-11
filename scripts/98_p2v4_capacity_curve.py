"""P2 v4, T1: end-to-end capacity curve of the cascade, measured instead of computed from the formula.

For the CICIoMT2024 6-class and 19-class cascades (student shield_s0 fp32, teacher_s0, the models of scripts/77), on one
pinned core with one thread:
  separate costs: student throughput mu_s and teacher throughput mu_t (batch 256 and batch 1), as scripts/77;
  end-to-end pipeline at escalation share e in {0, 2, 5, 10, 15, 20}%: the student scores the flows, the flows whose
  entropy exceeds the stream's (1 - e) quantile go to the teacher, and the whole pipeline is timed:
    batch 256: wall time per batch over 300 batches -> throughput;
    batch 1 (streaming): per-flow decision latency over 3,000 flows -> p50 / p95 / p99 and throughput = 1 / mean;
  formula prediction for the same e: 1 / (1/mu_s + e/mu_t) with the separately measured mu_s, mu_t.
Real test rows; escalated flows are the actual high-entropy flows of the stream. Background load at run time is
recorded (another CPU job may be active on other cores).
Output: outputs_v5/p2/capacity_curve.json."""
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from shield.data.datasets import load_processed
from shield.eval.latency import _PinnedCore
from shield.eval.predictions import load_student, run_dir
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import Paths, load_config

torch.set_num_threads(1)
OUT = Path("D:/shield_run/outputs_v5/p2")
ES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
CORE = 0


def ent(logits):
    z = logits - logits.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def main():
    P = Paths(False); tcfg = load_config("teacher", False); out = {"core": CORE, "cpu_percent_before": psutil.cpu_percent(1.0)}
    with _PinnedCore(CORE), torch.inference_mode():
        for ds, task in (("ciciomt", "category"), ("ciciomt", "attack")):
            data = load_processed(P.processed(ds), task); te = data.splits["test"]
            model, S = load_student(run_dir(P, ds, task, "shield", 0, tcfg["model"]), "cpu")
            teacher, _ = load_teacher(teacher_dir(P, ds, task, tcfg["model"], 0), torch.device("cpu"))
            n = 200_000
            Xf = np.ascontiguousarray(np.asarray(te.X[:n]), dtype=np.float32)
            Xs = np.ascontiguousarray(Xf if len(S) == Xf.shape[1] else Xf[:, S], dtype=np.float32)
            stu = lambda x: model(torch.from_numpy(x)).numpy()
            tea = lambda x: teacher(torch.from_numpy(x)).numpy().argmax(1)
            H = np.concatenate([ent(stu(Xs[i:i + 4096])) for i in range(0, n, 4096)])
            rng = np.random.default_rng(0)
            res = {"separate": {}, "curve": {}}
            for b, reps in ((256, 300), (1, 3000)):          # separate costs
                idx = rng.integers(0, n - b, reps + 50); ts, tt = [], []
                for k, s in enumerate(idx):
                    t0 = time.perf_counter(); stu(Xs[s:s + b]); t1 = time.perf_counter(); tea(Xf[s:s + b]); t2 = time.perf_counter()
                    if k >= 50:
                        ts.append(t1 - t0); tt.append(t2 - t1)
                res["separate"][b] = {"mu_s": b / float(np.median(ts)), "mu_t": b / float(np.median(tt))}
            for e in ES:
                thr = np.inf if e == 0 else float(np.quantile(H, 1 - e))
                # batch 256 end-to-end
                idx = rng.integers(0, n - 256, 350); tb = []
                for k, s in enumerate(idx):
                    t0 = time.perf_counter()
                    h = ent(stu(Xs[s:s + 256])); m = h > thr
                    if m.any():
                        tea(Xf[s:s + 256][m])
                    if k >= 50:
                        tb.append(time.perf_counter() - t0)
                # batch 1 streaming
                idx = rng.integers(0, n, 3050); tl, esc = [], 0
                for k, s in enumerate(idx):
                    t0 = time.perf_counter()
                    h = ent(stu(Xs[s:s + 1]))[0]
                    if h > thr:
                        tea(Xf[s:s + 1]); esc += k >= 50
                    if k >= 50:
                        tl.append(time.perf_counter() - t0)
                tl = np.array(tl) * 1e3
                f = {b: 1.0 / (1.0 / res["separate"][b]["mu_s"] + e / res["separate"][b]["mu_t"]) for b in (256, 1)}
                res["curve"][str(e)] = {"b256_throughput": 256 / float(np.median(tb)), "b256_formula": f[256],
                                        "b1_throughput": 1e3 / float(tl.mean()), "b1_formula": f[1],
                                        "b1_p50_ms": float(np.percentile(tl, 50)), "b1_p95_ms": float(np.percentile(tl, 95)),
                                        "b1_p99_ms": float(np.percentile(tl, 99)), "b1_escalated_share": esc / 3000}
                c = res["curve"][str(e)]
                print(ds, task, f"e={e:.2f}", f"b256 {c['b256_throughput']:.0f}/s (formula {c['b256_formula']:.0f})",
                      f"b1 {c['b1_throughput']:.0f}/s (formula {c['b1_formula']:.0f}) p50 {c['b1_p50_ms']:.3f} "
                      f"p99 {c['b1_p99_ms']:.3f} ms esc {c['b1_escalated_share']:.3f}", flush=True)
            out[f"{ds}_{task}"] = res
    out["cpu_percent_after"] = psutil.cpu_percent(1.0)
    json.dump(out, open(OUT / "capacity_curve.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
