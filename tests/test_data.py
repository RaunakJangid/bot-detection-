import numpy as np
import pyarrow.parquet as pq

from shield.data.datasets import Batcher, load_processed
from shield.data.ingest import BOTIOT_DROP, ciciomt_label, ingest_botiot, ingest_ciciomt, ingest_ciciot
from shield.data.preprocess import (NumericScaler, cap_per_class, ciciomt_category, dedup_mask, preprocess_botiot,
                                    preprocess_ciciomt, preprocess_ciciot, preprocess_dataset, shared_class,
                                    stratified_split)
from shield.data.registry import Registry


def test_ciciot_pipeline(tmp_path, ciciot_zip, data_cfg):
    counts = ingest_ciciot(ciciot_zip, tmp_path / "i", chunksize=100)
    assert counts == {"train": 400, "validation": 120, "test": 120}  # streaming keeps every row
    meta = preprocess_ciciot(tmp_path / "i", tmp_path / "p", data_cfg, drop=["IAT"])
    assert meta["tasks"]["family"] == ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
    assert "IAT" not in meta["features"]  # strict protocol
    assert meta["dedup"]["test_in"] - meta["dedup"]["test_out"] == 1  # planted train row removed from test
    d = load_processed(tmp_path / "p", "family")
    assert np.isfinite(d.splits["train"].X).all()
    assert abs(float(d.splits["train"].X.mean())) < 1e-3  # standardised with train statistics
    X = np.concatenate([d.splits[s].X for s in ("train", "validation", "test")])
    assert len(np.unique(X, axis=0)) == len(X)
    b = load_processed(tmp_path / "p", "binary").splits["train"].y
    assert set(np.unique(b)) == {0, 1}


def test_ciciomt_pipeline_and_shared(tmp_path, ciciot_zip, ciciomt_zip, data_cfg):
    counts = ingest_ciciomt(ciciomt_zip, tmp_path / "mt", chunksize=50)
    assert counts == {"train": 7 * 60, "test": 7 * 20}
    labels = set(pq.read_table(tmp_path / "mt" / "train.parquet", columns=["label"]).to_pandas()["label"])
    assert labels == {"Benign", "DDoS-ICMP", "DoS-SYN", "MQTT-DDoS-Connect_Flood", "Recon-Port_Scan", "ARP_Spoofing"}
    meta = preprocess_ciciomt(tmp_path / "mt", tmp_path / "pmt", data_cfg, drop=["IAT"])
    assert meta["tasks"]["category"] == ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
    assert meta["n_rows"]["validation"] > 0 and "IGMP" in meta["features"]

    ingest_ciciot(ciciot_zip, tmp_path / "ct", chunksize=100)
    interims = {"ciciot": tmp_path / "ct", "ciciomt": tmp_path / "mt"}
    m = preprocess_dataset("xciciomt", {"source": "ciciomt", "shared": True}, ["IAT"], interims, tmp_path / "x", data_cfg)
    assert m["features"] == ["Header_Length", "Protocol Type", "Rate", "Tot sum"]  # common, CICIoT order, no IAT/IGMP
    assert m["tasks"]["shared5"] == ["Benign", "DDoS", "DoS", "Recon", "Spoofing"]
    assert "MQTT-DDoS-Connect_Flood" not in m["labels"]  # MQTT has no CICIoT2023 counterpart


def test_label_maps():
    assert ciciomt_label("CICIoMT2024/train/TCP_IP-DDoS-UDP7_train.pcap.csv") == "DDoS-UDP"
    assert ciciomt_label("CICIoMT2024\\test\\Recon-VulScan_test.pcap.csv") == "Recon-VulScan"
    assert ciciomt_category("MQTT-DDoS-Publish_Flood") == "MQTT"
    assert shared_class("ciciot", "MITM-ArpSpoofing") == "Spoofing" and shared_class("ciciot", "Mirai-udpplain") is None
    assert shared_class("ciciomt", "ARP_Spoofing") == "Spoofing"


def test_botiot_pipeline(tmp_path, botiot_zip, data_cfg):
    counts = ingest_botiot(botiot_zip, tmp_path / "i", chunksize=100, workers=1)
    assert sum(counts.values()) == 900
    cols = pq.ParquetFile(next((tmp_path / "i").glob("part_*.parquet"))).schema.names
    assert not set(BOTIOT_DROP) & set(cols)  # leaky identity/time columns never reach disk
    assert "subcategory" in cols
    meta = preprocess_botiot(tmp_path / "i", tmp_path / "p", data_cfg, drop=["seq"])
    assert not set(BOTIOT_DROP) & set(meta["features"]) and "seq" not in meta["features"]
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
    two = stratified_split(y, (0.85, 0.15, 0.0), rng)  # validation carved from train: nothing lost
    assert len(two["test"]) == 0 and len(two["train"]) + len(two["validation"]) == len(y)


def test_registry():
    reg = Registry()
    assert reg.resolve("main") == ["ciciot", "ciciomt", "botiot"]
    assert reg.dropped("ciciot") == ["IAT"] and reg.dropped("ciciot_std") == []
    assert reg.dropped("botiot") == ["seq"]
    assert reg.for_stage("latency", reg.names) == ["ciciot", "ciciomt", "botiot"]
    assert reg.seeds("ciciot_std", [0, 1, 2, 3, 4]) == [0] and reg.seeds("ciciot", [0, 1]) == [0, 1]
    assert reg.resolve("ciciot,cross") == ["ciciot", "xciciot", "xciciomt"]


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
