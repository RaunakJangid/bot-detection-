import numpy as np
import pyarrow.parquet as pq

from shield.data.datasets import Batcher, load_processed
from shield.data.ingest import BOTIOT_DROP, ingest_botiot, ingest_ciciot
from shield.data.preprocess import (NumericScaler, cap_per_class, dedup_mask, preprocess_botiot,
                                    preprocess_ciciot, stratified_split)


def test_ciciot_pipeline(tmp_path, ciciot_zip, data_cfg):
    counts = ingest_ciciot(ciciot_zip, tmp_path / "i", chunksize=100)
    assert counts == {"train": 400, "validation": 120, "test": 120}  # streaming keeps every row
    meta = preprocess_ciciot(tmp_path / "i", tmp_path / "p", data_cfg)
    assert meta["tasks"]["family"] == ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
    d = load_processed(tmp_path / "p", "family")
    assert np.isfinite(d.splits["train"].X).all()
    assert abs(float(d.splits["train"].X.mean())) < 1e-3  # standardised with train statistics
    b = load_processed(tmp_path / "p", "binary").splits["train"].y
    assert set(np.unique(b)) == {0, 1}


def test_botiot_pipeline(tmp_path, botiot_zip, data_cfg):
    counts = ingest_botiot(botiot_zip, tmp_path / "i", chunksize=100, workers=1)
    assert sum(counts.values()) == 900
    cols = pq.ParquetFile(next((tmp_path / "i").glob("part_*.parquet"))).schema.names
    assert not set(BOTIOT_DROP) & set(cols)  # leaky identity/time columns never reach disk
    assert "subcategory" in cols
    meta = preprocess_botiot(tmp_path / "i", tmp_path / "p", data_cfg)
    assert not set(BOTIOT_DROP) & set(meta["features"])
    assert meta["dedup"]["conflicting_hashes"] >= 1 and meta["dedup"]["duplicate_rows_dropped"] >= 3
    d = load_processed(tmp_path / "p", "category")
    X = np.concatenate([d.splits[s].X for s in ("train", "validation", "test")])
    assert len(np.unique(X, axis=0)) == len(X)  # de-duplicated, so no row is shared across splits
    assert sum(meta["n_rows"].values()) == meta["dedup"]["rows_out"]


def test_dedup_mask_conflicts():
    h = np.array([5, 5, 7, 7, 9], dtype=np.uint64)
    lab = np.array([1, 1, 1, 2, 3])
    keep, stats = dedup_mask(h, lab)
    assert keep.tolist() == [True, False, False, False, True]
    assert stats["conflicting_rows_dropped"] == 2


def test_scaler_uses_train_only():
    rng = np.random.default_rng(0)
    train, test = rng.normal(5, 2, (1000, 3)).astype(np.float32), rng.normal(50, 9, (200, 3)).astype(np.float32)
    s = NumericScaler("none").fit(train)
    np.testing.assert_allclose(s.mean, train.mean(0), rtol=1e-4)
    out = np.empty_like(test)
    s.apply(test, out)
    assert out.mean() > 5  # test data is not re-centred on itself


def test_split_and_cap():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), [100, 20, 3])
    parts = stratified_split(y, (0.7, 0.15, 0.15), rng)
    allidx = np.concatenate(list(parts.values()))
    assert len(allidx) == len(np.unique(allidx)) == len(y)
    capped = cap_per_class(np.arange(len(y)), y, 10, rng)
    assert np.bincount(y[capped]).tolist() == [10, 10, 3]


def test_batcher_balanced_and_extras():
    import torch
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([0] * 18 + [1] * 2)
    extra = np.arange(20, dtype=np.float32)[:, None]
    b = Batcher(X, y, 8, torch.device("cpu"), balance_power=1.0, samples_per_epoch=4000, extras=[extra],
                on_device=False)
    ys, ok = [], True
    for xb, yb, eb in b:
        ys.append(yb.numpy())
        ok &= bool(torch.equal(xb[:, 0] / 2, eb[:, 0]))  # rows stay aligned across arrays
    frac = np.concatenate(ys).mean()
    assert ok and 0.4 < frac < 0.6
