"""Reproduction of a strong published CICIoMT2024 design (github.com/duaabn/CICIoMT2024: two-stage CatBoost,
benign-vs-attack gate then attack-class model; reported 90.39 macro-F1, 6-class, IAT kept, no cross-split de-dup)
under our protocols: `ciciomt` (strict: IAT removed, de-dup) and `ciciomt_std` (IAT kept, de-dup).
The gate threshold is chosen on VALIDATION macro-F1; test scored once."""

import numpy as np
from catboost import CatBoostClassifier

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.eval.metrics import classification_metrics
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("two_stage")


def cb(loss, seed=0):
    return CatBoostClassifier(loss_function=loss, iterations=3000, learning_rate=0.1, depth=8, task_type="GPU",
                              random_seed=seed, od_type="Iter", od_wait=150, verbose=False)


def main():
    p = parser(__doc__)
    args = p.parse_args()
    paths = Paths(args.smoke)
    for ds in ("ciciomt", "ciciomt_std"):
        for task in ("category", "attack"):
            data = load_processed(paths.processed(ds), task)
            tr, va, te = data.splits["train"], data.splits["validation"], data.splits["test"]
            ben = data.classes.index("Benign")
            rng = np.random.default_rng(0)
            es = stratified_sample(va.y, 300_000, rng, 100)
            Xtr, Xv, Xt = np.asarray(tr.X), np.asarray(va.X), np.asarray(te.X)
            gate = cb("Logloss").fit(Xtr, (tr.y != ben).astype(int), eval_set=(Xv[es], (va.y[es] != ben).astype(int)))
            att = tr.y != ben
            clf = cb("MultiClass").fit(Xtr[att], tr.y[att], eval_set=(Xv[es][va.y[es] != ben], va.y[es][va.y[es] != ben]))
            cls = clf.classes_.astype(int)

            def predict(X, thr):
                pa = gate.predict_proba(X)[:, 1]
                pc = clf.predict_proba(X)
                pred = cls[pc.argmax(1)]
                pred[pa < thr] = ben
                return pred

            onehot = lambda y: np.eye(data.n_classes, dtype=np.float32)[y]
            f1v = {t: classification_metrics(va.y, onehot(predict(Xv, t)), data.n_classes)["macro_f1"]
                   for t in np.round(np.arange(0.1, 0.95, 0.05), 2)}
            thr = max(f1v, key=f1v.get)
            m = classification_metrics(te.y, onehot(predict(Xt, thr)), data.n_classes, data.classes, full=True)
            save_json({"threshold": float(thr), "val_macro_f1": f1v[thr], "test": m}, paths.v3 / "two_stage" / f"{ds}_{task}.json")
            log.info("%s %s two-stage CatBoost: thr %.2f val %.4f test macro-F1 %.4f acc %.4f", ds, task, thr, f1v[thr],
                     m["macro_f1"], m["accuracy"])


if __name__ == "__main__":
    main()
