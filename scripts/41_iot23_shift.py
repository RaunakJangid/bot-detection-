"""E3: second capture-shift dataset (IoT-23, Stratosphere Lab; real IoT devices, 20 malware + 3 benign captures).

Question: does the false-alarm inflation seen on CICIoMT2024 (validation captures -> new captures) replicate on an
independent dataset with real devices, and does recalibration with a TIME-ORDERED burn-in restore the budget?
Design (strict protocol, fixed before the run):
  * Data: Zeek conn.log.labeled of every scenario (iot_23_datasets_small.tar.gz), read as a stream. Per capture at
    most --cap benign and --cap malicious flows (Bernoulli sampling with a rate from the file size), timestamps kept.
  * Features: Zeek flow statistics only (duration, bytes, packets, missed bytes, proto, service, conn_state, history
    letter counts, bytes per packet). Addresses, ports, uid and timestamps are NOT features (host/port shortcuts).
  * Shift split: captures are assigned to train or test with a fixed seed (honeypot captures 4-1 and 5-1 in train,
    7-1 in test, so benign traffic of unseen devices is in test); validation = 15% random slice of train captures.
    Control: a random row split of the pooled sample (70/15/15). Exact duplicate vectors removed across splits.
  * Models (binary benign / malicious): XGBoost, Random Forest, small MLP (5 seeds), small MLP distilled from the
    large MLP (5 seeds), large MLP (3 seeds), ensemble (mean of XGBoost and large MLP s0).
  * Metrics: validation vs test macro-F1; test FPR / recall at the 1% and 0.1% validation-benign thresholds; benign
    domain AUC; recalibration per test capture with the first K flows IN TIME: `clean` (first K benign flows,
    conformal quantile) and `corrected` (first K flows of any label, contamination-corrected, no labels)."""

import math
import re
import tarfile

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch import nn

from shield.cli import parser
from shield.data.datasets import stratified_sample
from shield.data.preprocess import dedup_across_splits, row_hash
from shield.eval.metrics import classification_metrics
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("iot23")
ARCHIVE = r"D:\bot-datasets\iot23\iot_23_datasets_small.tar.gz"
NUM = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
PROTO = ["tcp", "udp", "icmp"]
SERVICE = ["-", "dns", "http", "ssl", "dhcp", "irc", "ssh", "ftp"]
STATES = ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
HIST = list("ShADadFfRrTtWwIQ^")
# Malware family per capture (Stratosphere IoT-23 scenario table); honeypot captures are benign real devices.
FAMILY = {"1-1": "HideAndSeek", "3-1": "Muhstik", "7-1": "Mirai", "8-1": "Hakai", "9-1": "Hajime", "17-1": "Kenjiro",
          "20-1": "Torii", "21-1": "Torii", "33-1": "Kenjiro", "34-1": "Mirai", "35-1": "Mirai", "36-1": "Okiru",
          "39-1": "IRCBot", "42-1": "Trojan", "43-1": "Mirai", "44-1": "Mirai", "48-1": "Mirai", "49-1": "Mirai",
          "52-1": "Mirai", "60-1": "Gagfyt"}
COLS = ["ts", "uid", "oh", "op", "rh", "rp", "proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
        "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
        "resp_ip_bytes", "tunnel", "label", "detailed"]


def read_archive(cap, rng):
    rows, weights = [], {}
    with tarfile.open(ARCHIVE, "r|gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith("conn.log.labeled")):
                continue
            capture = re.search(r"(CTU-[A-Za-z]+-?[A-Za-z]*-Capture-[\d-]+)", m.name).group(1)
            est = max(m.size / 180, 1)
            rate = min(1.0, cap / est)
            f = tf.extractfile(m)
            kept = {"Benign": 0, "Malicious": 0}; seen = {"Benign": 0, "Malicious": 0}
            for raw in f:
                if raw[:1] == b"#":
                    continue
                lab = "Benign" if b"enign" in raw[-60:] else "Malicious"
                seen[lab] += 1
                # benign flows are rare in big malware captures: keep them at a higher rate
                r = 1.0 if (lab == "Benign" and kept[lab] < cap // 4) else rate
                if kept[lab] >= cap or rng.random() > r:
                    continue
                t = raw.decode("utf-8", "replace").split()
                if len(t) < 22:
                    continue
                rows.append([capture] + t[:23]); kept[lab] += 1
            log.info("%s: seen %s kept %s", capture, seen, kept)
            weights[capture] = {l: seen[l] / max(kept[l], 1) for l in seen}   # flows represented per kept flow
    df = pd.DataFrame(rows, columns=["capture"] + COLS[:23])
    df["w"] = [weights[c]["Malicious" if lab.lower().startswith("malicious") else "Benign"]
               for c, lab in zip(df.capture, df.label)]
    df["y"] = df["label"].str.lower().str.startswith("malicious").astype(np.int64)
    df["ts"] = df["ts"].astype(float)
    return df


def featurize(df):
    X = pd.DataFrame({c: pd.to_numeric(df[c].replace("-", np.nan), errors="coerce") for c in NUM})
    X["bpp_orig"] = X.orig_ip_bytes / X.orig_pkts.replace(0, np.nan)
    X["bpp_resp"] = X.resp_ip_bytes / X.resp_pkts.replace(0, np.nan)
    for v in PROTO:
        X[f"proto_{v}"] = (df.proto == v).astype(np.float32)
    for v in SERVICE:
        X[f"svc_{v}"] = (df.service == v).astype(np.float32)
    for v in STATES:
        X[f"st_{v}"] = (df.conn_state == v).astype(np.float32)
    h = df.history.fillna("-")
    for ch in HIST:
        X[f"h_{ch}"] = h.str.count(re.escape(ch)).astype(np.float32)
    X["h_len"] = h.str.len().astype(np.float32)
    A = X.to_numpy(np.float64)
    A = np.sign(A) * np.log1p(np.abs(A))
    return A, list(X.columns)


class MLP(nn.Module):
    def __init__(self, d, widths):
        super().__init__()
        layers, k = [], d
        for w in widths:
            layers += [nn.Linear(k, w), nn.ReLU()]
            k = w
        self.net = nn.Sequential(*layers, nn.Linear(k, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, Xv, yv, widths, seed, dev, teacher=None, epochs=8, T=2.0):
    torch.manual_seed(seed)
    m = MLP(X.shape[1], widths).to(dev)
    cnt = np.maximum(np.bincount(y, minlength=2), 1)
    ce = nn.CrossEntropyLoss(weight=torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
    soft = None
    if teacher is not None:
        with torch.no_grad():
            teacher.eval()
            soft = torch.cat([torch.softmax(teacher(Xt[i:i + 262144]) / T, 1) for i in range(0, len(Xt), 262144)])
    rng = np.random.default_rng(seed)
    best, state = -1, None
    for _ in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(y)), max(1, len(y) // 4096)):
            bi = torch.as_tensor(b, device=dev)
            z = m(Xt[bi])
            loss = ce(z, yt[bi])
            if soft is not None:
                loss = 0.5 * loss + 0.5 * T * T * nn.functional.kl_div(torch.log_softmax(z / T, 1), soft[bi],
                                                                       reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        s = mlp_score(m, Xv, dev)
        auc = roc_auc_score(yv, s) if len(np.unique(yv)) > 1 else 0
        if auc > best:
            best, state = auc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m


@torch.no_grad()
def mlp_score(m, X, dev):
    m.eval()
    return np.concatenate([torch.softmax(m(torch.tensor(X[i:i + 262144], device=dev)), 1)[:, 1].cpu().numpy()
                           for i in range(0, len(X), 262144)])


def conformal(s, alpha):
    k = int(math.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def evaluate(models, S, yv, yt, tst_df, K, alpha=0.01):
    out = {}
    for name, (sv, st) in S.items():
        r = {"val_macro_f1": classification_metrics(yv, np.log(np.clip(np.c_[1 - sv, sv], 1e-12, 1)), 2)["macro_f1"],
             "test_macro_f1": classification_metrics(yt, np.log(np.clip(np.c_[1 - st, st], 1e-12, 1)), 2)["macro_f1"]}
        for a in (0.01, 0.001):
            thr = np.quantile(sv[yv == 0], 1 - a)
            r[f"fpr@{a}"] = {"test_fpr": float((st[yt == 0] > thr).mean()), "test_recall": float((st[yt == 1] > thr).mean())}
        if tst_df is not None:   # time-ordered recalibration per test capture
            t_hi = np.quantile(sv[yv == 0], 1 - alpha / 10)
            rc = {"clean": [], "corrected": [], "fixed": []}
            thr = np.quantile(sv[yv == 0], 1 - alpha)
            for cap, g in tst_df.groupby("capture"):
                idx = g.sort_values("ts").index.to_numpy()
                ben = idx[yt[idx] == 0]
                if len(ben) < K + 500:
                    continue
                burn_c = ben[:K]; burn_m = idx[:K]
                ev = np.setdiff1d(idx, np.union1d(burn_c, burn_m))
                evb, eva = ev[yt[ev] == 0], ev[yt[ev] == 1]
                win = st[burn_m]; pi_hat = float((win > t_hi).mean())
                q = min(pi_hat + alpha * (1 - pi_hat), 0.999)
                hours = max((g.ts.max() - g.ts.min()) / 3600, 1e-9)
                wb = tst_df.w.to_numpy()[evb]
                for k, t in (("fixed", thr), ("clean", conformal(st[burn_c], alpha)), ("corrected", np.quantile(win, 1 - q))):
                    rc[k].append({"capture": cap, "fpr": float((st[evb] > t).mean()),
                                  "recall": float((st[eva] > t).mean()) if len(eva) else None, "n_benign": int(len(evb)),
                                  "hours": float(hours),
                                  "false_alarms_per_hour": float(wb[st[evb] > t].sum() / hours)})
            r["recalib"] = rc
        out[name] = r
    return out


def fit_all(X, y, Xv, yv, Xt, dev):
    S = {}
    rng = np.random.default_rng(0)
    it = stratified_sample(y, min(len(y), 2_000_000), rng, 1000)
    Xs, ys = X[it], y[it]
    for s in range(5):   # row/column subsampling so that seeds actually differ
        b = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                              early_stopping_rounds=50, random_state=s, subsample=0.8, colsample_bytree=0.8)
        b.fit(Xs, ys, eval_set=[(Xv, yv)], verbose=False)
        S[f"xgboost_s{s}"] = (b.predict_proba(Xv)[:, 1], b.predict_proba(Xt)[:, 1])
        rf = RandomForestClassifier(n_estimators=100, n_jobs=12, random_state=s).fit(np.nan_to_num(Xs[:1_000_000]), ys[:1_000_000])
        S[f"rf_s{s}"] = (rf.predict_proba(np.nan_to_num(Xv))[:, 1], rf.predict_proba(np.nan_to_num(Xt))[:, 1])
    Xn, Xvn, Xtn = (np.nan_to_num(a).astype(np.float32) for a in (Xs, Xv, Xt))
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xn, Xvn, Xtn = ((a - mu) / sd for a in (Xn, Xvn, Xtn))
    big = {}
    for s in range(3):
        big[s] = train_mlp(Xn, ys, Xvn, yv, (384, 384, 384, 384), s, dev)
        S[f"large_mlp_s{s}"] = (mlp_score(big[s], Xvn, dev), mlp_score(big[s], Xtn, dev))
    for s in range(5):
        m = train_mlp(Xn, ys, Xvn, yv, (64, 64), s, dev)
        S[f"small_mlp_scratch_s{s}"] = (mlp_score(m, Xvn, dev), mlp_score(m, Xtn, dev))
        m = train_mlp(Xn, ys, Xvn, yv, (64, 64), s, dev, teacher=big[0])
        S[f"small_mlp_kd_s{s}"] = (mlp_score(m, Xvn, dev), mlp_score(m, Xtn, dev))
    S["ensemble_s0"] = tuple(0.5 * (S["xgboost_s0"][i] + S["large_mlp_s0"][i]) for i in (0, 1))
    return S


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--cap", type=int, default=300_000)
    p.add_argument("--burn", type=int, default=1000)
    args = p.parse_args()
    P = Paths(args.smoke)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = P.v3 / "iot23"; out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "sample.parquet"
    rng = np.random.default_rng(0)
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = read_archive(args.cap, rng)
        df.to_parquet(cache)
    A, names = featurize(df)
    y = df.y.to_numpy()
    caps = sorted(df.capture.unique())
    mal = [c for c in caps if "Malware" in c]
    order = np.random.default_rng(20261011).permutation(mal)
    test_caps = set(order[: len(mal) // 3]) | {c for c in caps if c.startswith("CTU-Honeypot-Capture-7")}
    # same-family split: from every family with >= 2 captures hold out ~1/3 of its captures (at least one stays in
    # train), so test malware families are all seen in training; separates benign drift from unseen attacks
    fam_of = lambda c: FAMILY[c.split("Capture-")[1]]
    r3 = np.random.default_rng(20261012)
    same_caps = {c for c in caps if c.startswith("CTU-Honeypot-Capture-7")}
    for f in sorted({fam_of(c) for c in mal}):
        cs = sorted(c for c in mal if fam_of(c) == f)
        if len(cs) >= 2:
            same_caps |= set(r3.permutation(cs)[: max(1, len(cs) // 3)])
    res = {"captures": {c: {"n": int((df.capture == c).sum()), "malicious": int(y[df.capture.values == c].sum()),
                            "split": "test" if c in test_caps else "train"} for c in caps}, "features": names}
    log.info("test captures (unseen families): %s", sorted(test_caps))
    log.info("test captures (same families): %s", sorted(same_caps))
    res["same_family_test_captures"] = sorted(same_caps)
    res["family"] = {c: (fam_of(c) if "Malware" in c else "benign") for c in caps}

    for design in ("capture_shift", "same_family", "random_control"):
        r2 = np.random.default_rng(1)
        if design in ("capture_shift", "same_family"):
            is_test = df.capture.isin(test_caps if design == "capture_shift" else same_caps).to_numpy()
            tr_all = np.flatnonzero(~is_test)
            vmask = r2.random(len(tr_all)) < 0.15
            parts = {"train": tr_all[~vmask], "validation": tr_all[vmask], "test": np.flatnonzero(is_test)}
        else:
            u = r2.random(len(df))
            parts = {"train": np.flatnonzero(u < 0.7), "validation": np.flatnonzero((u >= 0.7) & (u < 0.85)),
                     "test": np.flatnonzero(u >= 0.85)}
        h = {s: row_hash(pd.DataFrame(np.nan_to_num(A[i], nan=-9.0))) for s, i in parts.items()}
        keep, _ = dedup_across_splits(h, {s: y[i] for s, i in parts.items()})
        parts = {s: i[keep[s]] for s, i in parts.items()}
        X, Xv, Xt = (A[parts[s]].astype(np.float32) for s in ("train", "validation", "test"))
        yy, yv, yt = (y[parts[s]] for s in ("train", "validation", "test"))
        log.info("%s: train %d (mal %.2f) val %d test %d (benign %d)", design, len(yy), yy.mean(), len(yv), len(yt),
                 int((yt == 0).sum()))
        S = fit_all(X, yy, Xv, yv, Xt, dev)
        tst = df.iloc[parts["test"]][["capture", "ts", "w"]].reset_index(drop=True) if design != "random_control" else None
        # per-flow scores for the streaming recalibration study (scripts/42_e8_stream.py)
        meta_t = df.iloc[parts["test"]]
        np.savez_compressed(out_dir / f"scores_{design}.npz", models=np.array(list(S)),
                            sv=np.stack([S[m][0] for m in S]), st=np.stack([S[m][1] for m in S]), yv=yv, yt=yt,
                            capture=meta_t.capture.to_numpy().astype(str), ts=meta_t.ts.to_numpy(),
                            w=meta_t.w.to_numpy(), Xv=np.nan_to_num(Xv).astype(np.float32),
                            Xt=np.nan_to_num(Xt).astype(np.float32), features=np.array(names))
        # benign domain AUC (validation benign vs test benign)
        import lightgbm as lgb
        Bv, Bt = np.nan_to_num(Xv[yv == 0]), np.nan_to_num(Xt[yt == 0])
        k = min(len(Bv), len(Bt), 50_000)
        M = np.vstack([Bv[r2.choice(len(Bv), k, replace=False)], Bt[r2.choice(len(Bt), k, replace=False)]])
        lab = np.r_[np.zeros(k), np.ones(k)]; fold = r2.integers(0, 2, 2 * k); sc = np.zeros(2 * k)
        for f in (0, 1):
            sc[fold == f] = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(M[fold != f], lab[fold != f]).predict_proba(M[fold == f])[:, 1]
        res[design] = {"benign_domain_auc": float(roc_auc_score(lab, sc)), "n": {s: int(len(i)) for s, i in parts.items()},
                       "models": evaluate(None, S, yv, yt, tst, args.burn)}
        fp = {m: round(100 * v["fpr@0.01"]["test_fpr"], 2) for m, v in res[design]["models"].items()}
        log.info("%s: benign domain AUC %.3f | test FPR@1%% target %s", design, res[design]["benign_domain_auc"], fp)
        save_json(res, out_dir / "iot23_shift.json")


if __name__ == "__main__":
    main()
