import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from shield.data.ingest import BOTIOT_CAT, BOTIOT_DROP, BOTIOT_NUM

CIC_FEATURES = ["flow_duration", "Header_Length", "Protocol Type", "Rate", "Tot sum", "IAT"]
CIC_LABELS = ["BenignTraffic", "DDoS-SYN_Flood", "DoS-UDP_Flood", "Mirai-udpplain", "Recon-PortScan",
              "DNS_Spoofing", "SqlInjection", "DictionaryBruteForce"]


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def ciciot_zip(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "cic.zip"
    frames = {}
    for split, n in (("train", 400), ("validation", 120), ("test", 120)):
        df = pd.DataFrame(rng.lognormal(2, 2, (n, len(CIC_FEATURES))), columns=CIC_FEATURES)
        df.loc[0, "Rate"] = np.inf
        df["label"] = [CIC_LABELS[i % len(CIC_LABELS)] for i in range(n)]
        frames[split] = df
    # Plant a train row in test (same features + label): cross-split de-dup must remove it from test.
    frames["test"].loc[2] = frames["train"].loc[2]
    with zipfile.ZipFile(path, "w") as zf:
        for split, df in frames.items():
            zf.writestr(f"CICIOT23/{split}/{split}.csv", _csv_bytes(df))
    return path


CIOMT_FEATURES = ["Header_Length", "Protocol Type", "Rate", "Tot sum", "IAT", "IGMP"]
CIOMT_FILES = ["Benign", "TCP_IP-DDoS-ICMP1", "TCP_IP-DDoS-ICMP2", "TCP_IP-DoS-SYN", "MQTT-DDoS-Connect_Flood",
               "Recon-Port_Scan", "ARP_Spoofing"]


@pytest.fixture
def ciciomt_zip(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "ciomt.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for split, n in (("train", 60), ("test", 20)):
            for name in CIOMT_FILES:
                df = pd.DataFrame(rng.lognormal(2, 2, (n, len(CIOMT_FEATURES))), columns=CIOMT_FEATURES)
                zf.writestr(f"CICIoMT2024/{split}/{name}_{split}.pcap.csv", _csv_bytes(df))
    return path


@pytest.fixture
def botiot_zip(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "bot.zip"
    cats = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]
    subs = {"Normal": "Normal", "DDoS": "TCP", "DoS": "UDP", "Reconnaissance": "Service_Scan", "Theft": "Keylogging"}
    with zipfile.ZipFile(path, "w") as zf:
        for f in range(1, 4):
            n = 300
            cat = [cats[i % 5] for i in range(n)]
            df = pd.DataFrame({c: rng.integers(0, 50, n).astype(float) for c in BOTIOT_NUM})
            for c in BOTIOT_DROP:
                df[c] = rng.integers(0, 1000, n)
            df["flgs"] = rng.choice(["e", "e s"], n)
            df["proto"] = rng.choice(["tcp", "udp", "arp"], n)
            df["state"] = rng.choice(["CON", "RST", "INT"], n)
            df["attack"] = [0 if c == "Normal" else 1 for c in cat]
            df["category"] = cat
            df["subcategory "] = [subs[c] for c in cat]  # trailing space as in the real files
            # Exact duplicates of row 0 (same label) and a conflicting duplicate of row 1.
            df.loc[n - 1, BOTIOT_NUM + BOTIOT_CAT] = df.loc[0, BOTIOT_NUM + BOTIOT_CAT].to_numpy()
            df.loc[n - 1, ["attack", "category", "subcategory "]] = df.loc[0, ["attack", "category", "subcategory "]].to_numpy()
            df.loc[n - 2, BOTIOT_NUM + BOTIOT_CAT] = df.loc[1, BOTIOT_NUM + BOTIOT_CAT].to_numpy()
            df.loc[n - 2, ["attack", "category", "subcategory "]] = [0, "Normal", "Normal"]
            zf.writestr(f"data_{f}.csv", _csv_bytes(df))
        zf.writestr("data_names.csv", "x\n")
    return path


@pytest.fixture
def data_cfg():
    return {"seed": 0, "transform": "signed_log", "botiot_split": [0.7, 0.15, 0.15], "min_category_count": 5,
            "max_per_class": None, "ciciomt_val_frac": 0.15}
