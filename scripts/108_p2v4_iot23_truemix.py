"""P2 v4: IoT-23 results under the capture's actual traffic distribution (evidence check, 49-1 in particular).

The test sets of scripts/41 pass two stages: (1) the archive sampler (per capture, all benign up to a cap, attack flows
Bernoulli-sampled; weight w = seen / kept per label), (2) de-duplication over train -> validation -> test (a feature vector
already seen earlier is dropped). Stage 2 removes repeated identical vectors, which for attack traffic is most of it.
Earlier P2 numbers used either no weights (distinct vectors) or only the stage-1 weights (X2 "true mix").
Here the split of scripts/41 is rebuilt exactly (verified against the stored score files: row count, labels and
timestamps per capture), and each test vector gets a volume weight W = sum of the sampler weights of all sampled flows
of that capture with the same feature vector (= estimated raw flows it stands for). Flows whose vector was already seen
in train/validation are not in the test set; their share is reported as lost coverage.
Per capture, three weightings: distinct (unweighted), sampler-only (stage-1 weights), volume (W). Per weighting:
weighted per-class precision, recall, support, macro-F1 of students small_mlp_kd_s0-4 and experts ensemble_s0-2
(15 pairs: expert - student), and the student's weighted escalated share at the validation 90th percentile.
Output: outputs_v5/p2/iot23_truemix.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("D:/shield_run/outputs_v5/p2")
CAPS = {"capture_shift": ["CTU-IoT-Malware-Capture-1-1", "CTU-IoT-Malware-Capture-35-1"],
        "same_family": ["CTU-IoT-Malware-Capture-49-1", "CTU-IoT-Malware-Capture-7-1"]}


def wmetrics(y, p, w):
    out = {}
    for c in (0, 1):
        tp = w[(p == c) & (y == c)].sum(); fp = w[(p == c) & (y != c)].sum(); fn = w[(p != c) & (y == c)].sum()
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0; rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        out[c] = {"precision": float(prec), "recall": float(rec), "f1": float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0,
                  "support": float(w[y == c].sum())}
    present = [c for c in (0, 1) if out[c]["support"] > 0 or w[p == c].sum() > 0]
    return {"per_class": out, "macro_f1": 100 * float(np.mean([out[c]["f1"] for c in present]))}


def ent2(q):
    q = np.clip(q, 1e-12, 1 - 1e-12); return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def main():
    from shield.data.preprocess import dedup_across_splits, row_hash
    from shield.utils.config import Paths
    P = Paths(False)
    s = importlib.util.spec_from_file_location("i23", Path(__file__).with_name("41_iot23_shift.py"))
    i23 = importlib.util.module_from_spec(s); s.loader.exec_module(i23)
    df = pd.read_parquet(P.v3 / "iot23" / "sample.parquet"); A, _ = i23.featurize(df); y = df.y.to_numpy().astype(np.int64)
    meta = json.load(open(P.v3 / "iot23" / "iot23_shift.json"))
    caps = sorted(df.capture.unique()); mal = [c for c in caps if "Malware" in c]
    order = np.random.default_rng(20261011).permutation(mal)
    test_caps = set(order[: len(mal) // 3]) | {c for c in caps if c.startswith("CTU-Honeypot-Capture-7")}
    same_caps = set(meta["same_family_test_captures"])
    H_all = row_hash(pd.DataFrame(np.nan_to_num(A, nan=-9.0)))
    w_all = df.w.to_numpy().astype(float); cap_all = df.capture.to_numpy()
    res = {}
    for design in ("capture_shift", "same_family"):
        r2 = np.random.default_rng(1)
        is_test = df.capture.isin(test_caps if design == "capture_shift" else same_caps).to_numpy()
        tr_all = np.flatnonzero(~is_test); vmask = r2.random(len(tr_all)) < 0.15
        parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
        h = {k: H_all[i] for k, i in parts.items()}
        keep, _ = dedup_across_splits(h, {k: y[i] for k, i in parts.items()})
        test_idx = parts["test"][keep["test"]]
        Z = np.load(P.v3 / "iot23" / f"scores_{design}.npz", allow_pickle=True); models = list(Z["models"])
        for cap in CAPS[design]:
            ti = test_idx[cap_all[test_idx] == cap]; m = Z["capture"] == cap
            verified = bool(len(ti) == m.sum() and np.array_equal(y[ti], Z["yt"][m].astype(np.int64))
                            and np.allclose(df.ts.to_numpy()[ti].astype(float), Z["ts"][m].astype(float)))
            rows_cap = np.flatnonzero(cap_all == cap)
            wsum = pd.Series(w_all[rows_cap]).groupby(H_all[rows_cap]).sum()
            W = wsum.reindex(H_all[ti]).to_numpy()
            weights = {"distinct": np.ones(len(ti)), "sampler_only": w_all[ti], "volume": W}
            yt = y[ti]
            st = lambda name: Z["st"][models.index(name)][m]
            PS = [(st(f"small_mlp_kd_s{s_}") > 0.5).astype(np.int64) for s_ in range(5)]
            PE = [((0.5 * (st(f"xgboost_s{j}") + st(f"large_mlp_s{j}"))) > 0.5).astype(np.int64) for j in range(3)]
            esc = [ent2(st(f"small_mlp_kd_s{s_}")) > np.quantile(ent2(Z["sv"][models.index(f"small_mlp_kd_s{s_}")]), 0.9) for s_ in range(5)]
            r = {"verified_against_score_file": verified, "n_test_vectors": int(len(ti)),
                 "raw_flows_capture_est": float(w_all[rows_cap].sum()), "raw_flows_covered_by_test_est": float(W.sum()),
                 "coverage": float(W.sum() / w_all[rows_cap].sum()), "weightings": {}}
            for wn, wv in weights.items():
                stu = [wmetrics(yt, p, wv) for p in PS]; exp = [wmetrics(yt, p, wv) for p in PE]
                diffs = [exp[j]["macro_f1"] - stu[s_]["macro_f1"] for s_ in range(5) for j in range(3)]
                r["weightings"][wn] = {"attack_share": float(wv[yt == 1].sum() / wv.sum()),
                                       "student_macro_f1": [round(x["macro_f1"], 2) for x in stu],
                                       "expert_macro_f1": [round(x["macro_f1"], 2) for x in exp],
                                       "expert_minus_student_all15": [round(x, 2) for x in diffs],
                                       "diff_median": float(np.median(diffs)), "diff_range": [float(min(diffs)), float(max(diffs))],
                                       "favour": {"expert": sum(x > 0.05 for x in diffs), "student": sum(x < -0.05 for x in diffs),
                                                  "tie": sum(abs(x) <= 0.05 for x in diffs)},
                                       "student_attack_recall": [round(x["per_class"][1]["recall"], 4) for x in stu],
                                       "expert_attack_recall": [round(x["per_class"][1]["recall"], 4) for x in exp],
                                       "student_benign_precision": [round(x["per_class"][0]["precision"], 4) for x in stu],
                                       "expert_benign_precision": [round(x["per_class"][0]["precision"], 4) for x in exp],
                                       "escalated_share_student": [round(float(wv[e].sum() / wv.sum()), 4) for e in esc]}
            res[f"{design}|{cap}"] = r
            print(cap, "verified", verified, "n", len(ti), "coverage %.4f" % r["coverage"], flush=True)
            for wn, x in r["weightings"].items():
                print(f"   {wn:13s} attack {x['attack_share']:.4f} diff median {x['diff_median']:+.2f} range "
                      f"[{x['diff_range'][0]:+.2f},{x['diff_range'][1]:+.2f}] favour {x['favour']} esc {x['escalated_share_student']}", flush=True)
    json.dump(res, open(OUT / "iot23_truemix.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
