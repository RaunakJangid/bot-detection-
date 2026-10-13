"""E6 export: everything the Mininet emulation needs as plain numpy (no torch in WSL).
SHIELD student (CICIoT2023 full, 34 classes, seed 0) weights + selected features; entropy escalation thresholds for
budgets {0, 5, 10, 20, 57}% set on VALIDATION (57% = test escalation observed under CICIoMT capture shift);
a replay set of test flows (student inputs, labels, ens4+rule cloud decisions)."""
import numpy as np
import torch

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import softmax
from shield.eval.predictions import load_student, run_dir
from shield.utils.config import Paths


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    import os
    os.environ.setdefault("SHIELD_OUTPUTS_DIR", "D:/shield_run/outputs_full")
    P = Paths(args.smoke)
    d = load_processed(P.processed("ciciot_full"), "class34", splits=("validation", "test"))
    st, S = load_student(run_dir(P, "ciciot_full", "class34", "shield", 0), torch.device("cpu"))
    lin = [m for m in st.modules() if isinstance(m, torch.nn.Linear)]
    W = {f"W{i}": m.weight.detach().numpy().T.astype(np.float32) for i, m in enumerate(lin)}
    W.update({f"b{i}": m.bias.detach().numpy().astype(np.float32) for i, m in enumerate(lin)})
    lc = P.v3 / "logits" / "ciciot_full_class34"
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    pv = softmax(np.load(lc / "shield_s0_validation.npy").astype(np.float64))
    thr = {b: (float(np.quantile(ent(pv), 1 - b / 100)) if b > 0 else np.inf) for b in (0, 5, 10, 20, 57)}
    te = d.splits["test"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(stratified_sample(te.y, 60_000, rng, 50))
    Xs = np.asarray(te.X[idx])[:, S].astype(np.float32)
    cloud = np.load(lc / "ens4_s0_test.npy")[idx].argmax(1).astype(np.int16)
    out = r"D:\shield_run\outputs_v3\e6\e6_bundle.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=Xs, y=te.y[idx].astype(np.int16), cloud=cloud, n_layers=len(lin),
             budgets=np.array(list(thr)), thresholds=np.array(list(thr.values())), **W)
    print("exported", Xs.shape, "layers", len(lin), "thresholds", thr)


if __name__ == "__main__":
    main()
