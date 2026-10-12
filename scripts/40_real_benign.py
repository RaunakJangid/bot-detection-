"""E7: false alarms on REAL benign IoT traffic, in the CICIoT2023 feature space.

Problem: our detectors use the 46-column Kaggle CICIoT2023 CSVs, and no home capture exists. Solution:
  * Feature extraction: SentryIoT's extractor (MIT licence, github.com/stermiix/SentryIoT, codigo/captura/extrator.py),
    which reproduces every row of the authors' official 39-column CSVs on the pcaps it was checked on. The 39 and the
    46-column releases share 37 non-leaky columns (Kaggle "Duration" is the TTL = extractor "Time_To_Live"; IAT is
    dropped as in our strict protocol). Windows of 10 and of 100 consecutive frames (both are used in CICIoT2023).
  * Real benign traffic: UNSW IoT traces (Sivanathan et al., IEEE TMC 2019; 28 IoT devices plus phones/laptops in a
    real smart-home testbed, full days) and the IoT-23 honeypot captures (real Philips Hue, Amazon Echo and Somfy
    door-lock devices, Stratosphere Lab). Every window is benign by construction.
  * Detectors: binary benign/attack models retrained on the strict CICIoT2023 subset restricted to the 37 shared
    columns (XGBoost; small MLP x3 seeds; large MLP; ensemble = mean of XGBoost and large MLP). Threshold: 1% FPR on
    CICIoT validation benign flows (as everywhere else).
  * Reported per source and window size: realised FPR, false alarms per hour of traffic, the same after a
    time-ordered clean burn-in (first --burn windows of the first file, conformal quantile; evaluated on later
    traffic), the lab recall at each threshold, and a benign domain AUC (CICIoT benign vs real benign).
Limits: the 37-column models are not our 45-column production models; UNSW is a 2016 testbed; window membership
mixes devices exactly as in CICIoT2023's own pcap windows."""

import glob
import math
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch import nn

from shield.cli import parser
from shield.data.datasets import load_processed, stratified_sample
from shield.utils.config import Paths
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("realbenign")
SENTRY = r"D:\bot-datasets\SentryIoT"
RAW = r"D:\bot-datasets\realbenign"
RENAME = {"Time_To_Live": "Duration"}


def read_any(f):
    """(ts, frame) from a classic pcap (SentryIoT reader) or a pcapng stream (dpkt; the UNSW traces are pcapng)."""
    import io
    import dpkt
    from codigo.captura.extrator import ler_pcap
    head = f.read(4)
    f = io.BufferedReader(_Prefixed(head, f))
    if head == b"\x0a\x0d\x0d\x0a":
        for ts, buf in dpkt.pcapng.Reader(f):
            yield float(ts), buf
    else:
        yield from ler_pcap(f)


class _Prefixed(__import__("io").RawIOBase):
    """Stream that replays already-read bytes before the rest of a non-seekable stream."""

    def __init__(self, head, f):
        self.head, self.f = head, f

    def readable(self):
        return True

    def readinto(self, b):
        if self.head:
            n = min(len(b), len(self.head)); b[:n] = self.head[:n]; self.head = self.head[n:]
            return n
        data = self.f.read(len(b))
        b[:len(data)] = data
        return len(data)


def frames(path):
    sys.path.insert(0, SENTRY)
    if path.endswith(".tar.gz"):
        with tarfile.open(path, "r|gz") as tf:
            for m in tf:
                if m.isfile() and (m.name.endswith(".pcap") or m.name.endswith(".pcapng")):
                    yield from read_any(tf.extractfile(m))
    else:
        with open(path, "rb") as f:
            yield from read_any(f)


def extract(path, windows=(10, 100)):
    sys.path.insert(0, SENTRY)
    from codigo.captura.extrator import Extrator, medir_quadro
    ex = {w: Extrator(w) for w in windows}
    rows = {w: [] for w in windows}
    prev = None
    for ts, q in frames(path):
        m = medir_quadro(ts, q, prev)
        if m is not None:
            prev = ts
        for w, e in ex.items():
            r = e.acumular(m)
            if r is not None:
                r["ts_end"] = ts
                rows[w].append(r)
    return {w: pd.DataFrame(r).rename(columns=RENAME) for w, r in rows.items()}


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


def train_mlp(X, y, Xv, yv, widths, seed, dev, epochs=6):
    torch.manual_seed(seed)
    m = MLP(X.shape[1], widths).to(dev)
    w = torch.tensor((np.bincount(y, minlength=2).sum() / np.maximum(np.bincount(y, minlength=2), 1)) ** 0.5,
                     dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
    rng = np.random.default_rng(seed)
    best, state = -1, None
    for _ in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(y)), max(1, len(y) // 8192)):
            bi = torch.as_tensor(b, device=dev)
            opt.zero_grad(); ce(m(Xt[bi]), yt[bi]).backward(); opt.step()
        s = score(m, Xv, dev)
        auc = roc_auc_score(yv, s)
        if auc > best:
            best, state = auc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m


@torch.no_grad()
def score(m, X, dev):
    m.eval()
    out = [torch.softmax(m(torch.tensor(X[i:i + 262144], device=dev)), 1)[:, 1].cpu().numpy()
           for i in range(0, len(X), 262144)]
    return np.concatenate(out)


def conformal(s, alpha=0.01):
    k = int(math.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def main():
    p = parser(__doc__, datasets=False)
    p.add_argument("--burn", type=int, default=2000)
    # Columns whose Kaggle-46 values are not the per-window quantity the authors' extractor computes (benign
    # marginals differ by orders of magnitude; e.g. Kaggle Header_Length median 5e5 vs ~20 bytes, Variance ~0.9
    # while Std ~300). Decided from benign feature marginals only, never from detection results.
    p.add_argument("--exclude", nargs="*",
                   default=["Header_Length", "ack_count", "syn_count", "rst_count", "Number", "Variance"])
    p.add_argument("--tag", default="")
    args = p.parse_args()
    P = Paths(args.smoke)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = P.v3 / "realbenign"; feat_dir = out_dir / "features"; feat_dir.mkdir(parents=True, exist_ok=True)

    # 1. extract real benign windows (cached)
    sources = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.pcap")) + glob.glob(os.path.join(RAW, "*.tar.gz"))):
        name = os.path.basename(f).split(".")[0]
        if os.path.getsize(f) < 1_000_000:
            log.warning("skip %s (incomplete download)", name); continue
        cache = {w: feat_dir / f"{name}_w{w}.parquet" for w in (10, 100)}
        if not all(c.exists() for c in cache.values()):
            log.info("extracting %s", name)
            try:
                dfs = extract(f)
            except Exception as e:  # noqa: BLE001
                log.warning("failed %s: %s", name, e); continue
            for w, df in dfs.items():
                df.to_parquet(cache[w])
        sources[name] = {w: pd.read_parquet(c) for w, c in cache.items()}
        log.info("%s: %d windows (w10), %d (w100)", name, len(sources[name][10]), len(sources[name][100]))
    groups = {"unsw": [s for s in sources if s.startswith("unsw")],
              "iot23_honeypot": [s for s in sources if s.startswith("hp")]}

    # 2. detectors on the 37 shared columns, strict CICIoT subset, binary
    d = load_processed(P.processed("ciciot"), "binary")
    meta = __import__("json").load(open(P.processed("ciciot") / "meta.json"))
    feats = [c for c in d.features if c in next(iter(sources.values()))[10].columns and c not in args.exclude]
    ix = [d.features.index(c) for c in feats]
    log.info("%d shared columns: %s", len(feats), feats)
    sc = {k: np.asarray(meta["scaler"][k], np.float64)[ix] for k in ("median", "mean", "std")}

    def transform(df):
        R = df[feats].to_numpy(np.float64)
        R[~np.isfinite(R)] = np.nan
        R = np.where(np.isnan(R), sc["median"], R)
        return ((np.sign(R) * np.log1p(np.abs(R)) - sc["mean"]) / sc["std"]).astype(np.float32)

    tr, va, te = d.splits["train"], d.splits["validation"], d.splits["test"]
    ben = d.classes.index("Benign")
    rng = np.random.default_rng(0)
    it = stratified_sample(tr.y, 2_000_000, rng, 1000)
    X, y = np.asarray(tr.X[it])[:, ix], (tr.y[it] != ben).astype(np.int64)
    Xv, yv = np.asarray(va.X)[:, ix], (va.y != ben).astype(np.int64)
    Xte, yte = np.asarray(te.X)[:, ix], (te.y != ben).astype(np.int64)
    models = {}
    bst = xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.1, tree_method="hist", device="cuda",
                            early_stopping_rounds=50, random_state=0)
    bst.fit(X, y, eval_set=[(Xv[::4], yv[::4])], verbose=False)
    models["xgboost"] = lambda Z: bst.predict_proba(Z)[:, 1]
    for s in range(3):
        m = train_mlp(X, y, Xv[::4], yv[::4], (64, 64), s, dev)
        models[f"small_mlp_s{s}"] = (lambda mm: lambda Z: score(mm, Z, dev))(m)
    big = train_mlp(X, y, Xv[::4], yv[::4], (384, 384, 384, 384), 0, dev)
    models["large_mlp"] = lambda Z: score(big, Z, dev)
    models["ensemble"] = lambda Z: 0.5 * (models["xgboost"](Z) + models["large_mlp"](Z))

    # 3. evaluation
    res = {"shared_columns": feats, "models": {}}
    for name, fn in models.items():
        sv, st = fn(Xv), fn(Xte)
        thr = float(np.quantile(sv[yv == 0], 0.99))
        r = {"lab": {"val_fpr": float((sv[yv == 0] > thr).mean()), "test_fpr": float((st[yte == 0] > thr).mean()),
                     "test_recall": float((st[yte == 1] > thr).mean())}, "real": {}}
        for g, names in groups.items():
            for w in (10, 100):
                parts = [sources[n][w].assign(src=n) for n in names]
                if not parts:
                    continue
                df = pd.concat(parts, ignore_index=True).sort_values(["src", "ts_end"], kind="stable")
                Z = transform(df); s = fn(Z)
                hours = sum((sources[n][w].ts_end.max() - sources[n][w].ts_end.min()) / 3600 for n in names)
                fpr = float((s > thr).mean())
                # time-ordered clean burn-in: first `burn` windows of the first file in time
                first = df.src.values == names[0]
                b = np.flatnonzero(first)[:args.burn]
                rest = np.ones(len(s), bool); rest[b] = False
                t2 = float(conformal(s[b]))
                r["real"][f"{g}_w{w}"] = {
                    "windows": int(len(s)), "hours": float(hours),
                    "fpr_fixed": fpr, "alarms_per_hour_fixed": float(fpr * len(s) / max(hours, 1e-9)),
                    "fpr_burnin": float((s[rest] > t2).mean()),
                    "alarms_per_hour_burnin": float((s[rest] > t2).sum() / max(hours, 1e-9)),
                    "lab_recall_at_burnin_threshold": float((st[yte == 1] > t2).mean()),
                    "per_file_fpr_fixed": {n: float((s[df.src.values == n] > thr).mean()) for n in names}}
        res["models"][name] = r
        log.info("%s: lab test FPR %.4f recall %.4f | %s", name, r["lab"]["test_fpr"], r["lab"]["test_recall"],
                 {k: (round(100 * v["fpr_fixed"], 2), round(100 * v["fpr_burnin"], 2),
                      round(100 * v["lab_recall_at_burnin_threshold"], 2)) for k, v in r["real"].items()})
    # benign domain AUC (CICIoT benign test vs real benign), XGBoost domain classifier, 2-fold
    import lightgbm as lgb
    for g, names in groups.items():
        for w in (10, 100):
            if not names:
                continue
            A = transform(pd.concat([sources[n][w] for n in names]))
            B = Xte[yte == 0]
            k = min(len(A), len(B), 50_000)
            M = np.vstack([B[rng.choice(len(B), k, replace=False)], A[rng.choice(len(A), k, replace=False)]])
            lab = np.r_[np.zeros(k), np.ones(k)]; fold = rng.integers(0, 2, 2 * k); sc_ = np.zeros(2 * k)
            for f in (0, 1):
                c = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(M[fold != f], lab[fold != f])
                sc_[fold == f] = c.predict_proba(M[fold == f])[:, 1]
            res.setdefault("domain_auc", {})[f"{g}_w{w}"] = float(roc_auc_score(lab, sc_))
    log.info("domain AUC %s", res.get("domain_auc"))
    res["excluded_columns"] = args.exclude
    save_json(res, out_dir / f"real_benign{args.tag}.json")


if __name__ == "__main__":
    main()
