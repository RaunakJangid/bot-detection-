"""Constrained-CPU profile of the deployable detector (not a Raspberry Pi measurement): one core, one thread
(OMP/MKL/OpenBLAS = 1, process pinned to CPU 0), SHIELD student on CICIoT 34-class (full data) as fp32 PyTorch and
as the integer-only model (W8A8 numpy reference implementation). Reports per-flow latency at batch 1 and
throughput at batch 256, plus model size."""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import time

import numpy as np
import psutil
import torch

from shield.data.datasets import load_processed, stratified_sample
from shield.eval.predictions import load_student
from shield.models.quant import IntMLP, calibrate
from shield.utils.config import Paths

psutil.Process().cpu_affinity([0])
torch.set_num_threads(1)
paths = Paths()
run = paths.outputs.with_name("outputs_full") / "kd" / "ciciot_full_class34" / "shield_s0"
model, S = load_student(run, torch.device("cpu"))
data = load_processed(paths.processed("ciciot_full"), "class34", splits=("train", "test"))
X = np.ascontiguousarray(np.asarray(data.splits["test"].X[:20000])[:, S], dtype=np.float32)
cal = np.asarray(data.splits["train"].X[stratified_sample(data.splits["train"].y, 100000, np.random.default_rng(0), 50)])[:, S]
q = IntMLP(model, calibrate(model, cal))


def bench(fn, batch, reps):
    xs = [X[i:i + batch] for i in range(0, batch * reps, batch)]
    fn(xs[0])
    t0 = time.perf_counter()
    for x in xs:
        fn(x)
    dt = time.perf_counter() - t0
    return 1000 * dt / reps, batch * reps / dt


with torch.no_grad():
    f32 = lambda x: model(torch.from_numpy(x))
    for name, fn in (("fp32 torch", f32), ("int8 W8A8 (numpy)", q.forward)):
        l1, _ = bench(fn, 1, 2000)
        _, tp = bench(fn, 256, 60)
        print(f"{name}: latency batch1 {l1:.3f} ms/flow | throughput batch256 {tp:,.0f} flows/s")
print("size: fp32", 4 * sum(p.numel() for p in model.parameters()), "B | int8", q.size_bytes(), "B | features", len(S))
print("process RSS MB:", round(psutil.Process().memory_info().rss / 2**20))
