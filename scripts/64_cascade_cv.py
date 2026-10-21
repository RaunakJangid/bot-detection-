"""P2 revision: 5-fold cross-validation of the detector cascade's escalation threshold (cached model outputs).

The cascade's only fitted quantity on top of the trained models is the entropy threshold that escalates a planned
share b = 10% of validation flows. Here the validation set is split into 5 stratified folds; for each fold the
threshold is fitted on the other 4 folds and then
  * on the held-out validation fold: escalation share and cascade macro-F1 (in-distribution check);
  * on the shifted test traffic: escalation share e_test and cascade macro-F1 with that threshold (f1_planned).
The capped cascade (test-entropy quantile, label-free) does not depend on the fold and is reported once.
Settings and models as in 54_escalation_shift.py (students shield_s0-4 / small_mlp_kd_s0-4; expert teacher_s0 on
CICIoT/CICIoMT, ensemble_s0 on IoT-23). Output: <v3>/cascade_cv/{folds.csv, summary.csv}."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from shield.cli import parser
from shield.data.datasets import load_processed
from shield.eval.metrics import softmax
from shield.utils.config import Paths
from shield.utils.logging import get_logger

log = get_logger("cascade_cv")
B, FOLDS = 0.10, 5


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def f1(y, p, binary):
    return float(100 * f1_score(y, p, average="binary" if binary else "macro"))


def run_setting(name, shift, Pv, Pt, pe_v, pe_t, yv, yt, seed, binary):
    Hv, Ht = entropy(Pv), entropy(Pt)
    ps_v, ps_t = Pv.argmax(1), Pt.argmax(1)
    cap = np.where(Ht > np.quantile(Ht, 1 - B), pe_t, ps_t)
    f1_capped = f1(yt, cap, binary)
    rows = []
    for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(Hv, yv)):
        t = np.quantile(Hv[tr], 1 - B)
        esc_v = Hv[te] > t
        esc_t = Ht > t
        rows.append({"setting": name, "shift": shift, "seed": seed, "fold": f,
                     "val_escalated": 100 * esc_v.mean(),
                     "val_f1": f1(yv[te], np.where(esc_v, pe_v[te], ps_v[te]), binary),
                     "test_escalated": 100 * esc_t.mean(),
                     "test_f1_planned": f1(yt, np.where(esc_t, pe_t, ps_t), binary),
                     "test_f1_capped": f1_capped})
    return rows


def main():
    parser(__doc__, datasets=False).parse_args()
    P = Paths(False)
    od = P.v3 / "cascade_cv"; od.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, task in (("ciciot", "class34"), ("ciciomt", "binary"), ("ciciomt", "category"), ("ciciomt", "attack")):
        d = load_processed(P.processed(ds), task, splits=("validation", "test"))
        yv, yt = d.splits["validation"].y, d.splits["test"].y
        lc = P.v3 / "logits" / f"{ds}_{task}"
        pe_v = np.load(lc / "teacher_s0_validation.npy").argmax(1)
        pe_t = np.load(lc / "teacher_s0_test.npy").argmax(1)
        for s in range(5):
            Pv = softmax(np.load(lc / f"shield_s{s}_validation.npy").astype(np.float64))
            Pt = softmax(np.load(lc / f"shield_s{s}_test.npy").astype(np.float64))
            rows += run_setting(f"{ds}_{task}", ds != "ciciot", Pv, Pt, pe_v, pe_t, yv, yt, s, False)
        log.info("%s_%s done", ds, task)
    two = lambda p: np.c_[1 - p, p]
    for design in ("same_family", "capture_shift"):
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz")
        models = list(Z["models"]); yv, yt, capt = Z["yv"], Z["yt"], Z["capture"]
        ie = models.index("ensemble_s0")
        pe_v, pe_all = (Z["sv"][ie] > 0.5).astype(int), (Z["st"][ie] > 0.5).astype(int)
        for c in np.unique(capt):
            m = capt == c
            if (yt[m] == 0).sum() < 100 or (yt[m] == 1).sum() < 100:
                continue
            for s in range(5):
                i = models.index(f"small_mlp_kd_s{s}")
                rows += run_setting(f"iot23_{design}_{c}", True, two(Z["sv"][i]), two(Z["st"][i][m]), pe_v,
                                    pe_all[m], yv, yt[m], s, True)
            log.info("iot23 %s %s done", design, c)
    df = pd.DataFrame(rows)
    df.to_csv(od / "folds.csv", index=False)
    cols = ["val_escalated", "val_f1", "test_escalated", "test_f1_planned", "test_f1_capped"]
    summ = df.groupby("setting")[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(od / "summary.csv")
    log.info("5-fold CV (25 folds per setting = 5 seeds x 5 folds), mean / sd:\n%s", summ.to_string())


if __name__ == "__main__":
    main()
