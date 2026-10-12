"""Seed replicates for the tree models and tree ensembles on CICIoMT2024 (E2 gap: GBDT/ensembles had one seed).

Same settings as 19_detector_v4.py (LightGBM bagging 0.8, CatBoost GPU) with seeds 1-2 added to the existing s0,
XGBoost with row/column subsampling 0.8 (xgbsub_s0..4, so that seeds differ), and a seed-matched tree ensemble
treeens_s{k} = mean probability of xgbsub_s{k}, lgbm_s{k}, catboost_s{k}. Logits go to outputs_v3/logits, where
34_shift_and_base_rate.py, 36_recalibration.py and 37_fpr_variance.py pick them up."""

import importlib.util

import numpy as np
import xgboost as xgb

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("tree_seeds")
_s = importlib.util.spec_from_file_location("dv4", __file__.replace("43_tree_seeds", "19_detector_v4"))
dv4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(dv4)
SPLITS = ("validation", "test")


def run_xgbsub(data, seed, cap=6_000_000):
    tr, va = data.splits["train"], data.splits["validation"]
    idx, vidx = dv4.train_rows(tr.y, cap, seed), dv4.train_rows(va.y, 500_000, seed)
    n = data.n_classes
    clf = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=50, random_state=seed, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic" if n == 2 else "multi:softprob")
    clf.fit(np.asarray(tr.X[idx]), tr.y[idx], eval_set=[(np.asarray(va.X[vidx]), va.y[vidx])], verbose=False)
    out = {}
    for s in SPLITS:
        X = np.asarray(data.splits[s].X)
        p = np.vstack([clf.predict_proba(X[i:i + 500_000]) for i in range(0, len(X), 500_000)])
        out[s] = np.log(np.clip(p, 1e-12, 1))
    return out


def main():
    args = parser(__doc__, datasets=False).parse_args()
    P = Paths(args.smoke)
    for task in ("binary", "category", "attack"):
        ds = "ciciomt"
        data = load_processed(P.processed(ds), task)
        for seed in range(5):
            name = f"xgbsub_s{seed}"
            if not dv4.have(P, ds, task, name):
                dv4.save(P, ds, task, name, run_xgbsub(data, seed)); log.info("%s_%s %s done", ds, task, name)
        for seed in (1, 2):
            for name, fn in ((f"lgbm_s{seed}", lambda: dv4.run_lgbm(data, 6_000_000, seed)),
                             (f"catboost_s{seed}", lambda: dv4.run_catboost(data, 6_000_000, seed))):
                if not dv4.have(P, ds, task, name):
                    dv4.save(P, ds, task, name, fn()); log.info("%s_%s %s done", ds, task, name)
        for seed in range(3):
            name = f"treeens_s{seed}"
            if dv4.have(P, ds, task, name):
                continue
            parts = [dv4.load(P, ds, task, f"{m}_s{seed}") for m in ("xgbsub", "lgbm", "catboost")]
            dv4.save(P, ds, task, name, {s: np.log(np.clip(np.mean([softmax(z[s].astype(np.float64)) for z in parts], 0),
                                                           1e-12, 1)) for s in SPLITS})
            log.info("%s_%s %s done", ds, task, name)


if __name__ == "__main__":
    main()
