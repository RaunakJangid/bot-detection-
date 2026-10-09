"""Deep IDS baselines (CNN-LSTM, CNN-GRU, FT-Transformer) under the strict protocol: same splits, same training
loop and early stopping on VALIDATION macro-F1 as our teachers (teacher.yaml train settings, inverse-sqrt class
weights), test scored once. Logits -> outputs_v3/logits/<dataset>_<task>/<model>_s<seed>_{validation,test}.npy."""

import time

import numpy as np
import torch
from torch import nn

from shield.cli import parser
from shield.data.datasets import Batcher, class_weights, load_processed, predict_logits
from shield.engine import fit
from shield.eval.metrics import classification_metrics
from shield.models.deep import build_deep
from shield.models.student import count_params
from shield.models.train import train_cfg_for
from shield.utils.config import Paths, load_config
from shield.utils.device import get_device
from shield.utils.io import save_json
from shield.utils.logging import get_logger
from shield.utils.seed import seed_everything

log = get_logger("deep")
JOBS = [("ciciot_full", "class34"), ("ciciomt", "category"), ("ciciomt", "attack")]


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--models", nargs="*", default=["cnn_lstm", "cnn_gru", "ftt"])
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    paths, device = Paths(args.smoke), get_device(args.cpu)
    tcfg_all = load_config("teacher", args.smoke)
    for ds, task in JOBS:
        data = load_processed(paths.processed(ds), task)
        tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
        tcfg = {**train_cfg_for(tcfg_all, ds), "epochs": args.epochs, "batch_size": 4096, "amp": True,
                "samples_per_epoch": min(len(tr.y), 4_000_000)}
        cw = class_weights(tr.y, data.n_classes, "inverse_sqrt")
        ce = nn.CrossEntropyLoss(weight=torch.as_tensor(cw, device=device))
        for name in args.models:
            for seed in args.seeds:
                out = paths.v3 / "deep" / f"{ds}_{task}_{name}_s{seed}.json"
                lc = paths.v3 / "logits" / f"{ds}_{task}"
                if out.exists() and not args.force:
                    continue
                seed_everything(seed)
                model = build_deep(name, data.n_features, data.n_classes)
                batcher = Batcher(tr.X, tr.y, tcfg["batch_size"], device, samples_per_epoch=tcfg["samples_per_epoch"],
                                  seed=seed)
                t0 = time.time()
                fit(model, batcher, va.X, va.y, lambda m, b: ce(m(b[0]), b[1]), tcfg, device, data.n_classes,
                    f"{ds}_{task}_{name}_s{seed}")
                del batcher
                z = {s: predict_logits(model, data.splits[s].X, device, 32768, amp=True) for s in ("validation", "test")}
                lc.mkdir(parents=True, exist_ok=True)
                for s, v in z.items():
                    np.save(lc / f"{name}_s{seed}_{s}.npy", v.astype(np.float16))
                m = classification_metrics(te.y, z["test"], data.n_classes, data.classes, full=True)
                save_json({"params": count_params(model), "train_seconds": round(time.time() - t0),
                           "val_macro_f1": classification_metrics(va.y, z["validation"], data.n_classes)["macro_f1"],
                           "test": m}, out)
                log.info("%s %s %s s%d: test macro-F1 %.4f (params %d)", ds, task, name, seed, m["macro_f1"],
                         count_params(model))
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
