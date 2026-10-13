#!/usr/bin/env python3
"""E6: Mininet emulation of the edge-cloud cascade (run as root inside WSL Ubuntu; numpy only).

Topology: 20 IoT hosts -> s1 (OVS) -> gateway (16 KB SHIELD student, numpy, batch 1) -> s2 -> cloud host over a WAN
link (10 ms each way, 100 Mbit/s). IoT hosts replay CICIoT2023 test flows (student inputs) at an aggregate rate R
(UDP). The gateway decides every flow; flows whose entropy exceeds the validation threshold for budget b are escalated
to the cloud, which returns the ens4+rule decision after a fixed service time (the ensemble's compute is emulated with
--cloud-ms; its decisions are the real precomputed ones). Single-threaded gateway and cloud = FIFO queues.
Grid: R in {1k, 5k, 10k, 20k} flows/s x b in {0, 5, 10, 20, 57}% (57% = escalation observed under capture shift).
Per run: throughput, loss, achieved escalation, decision latency p50/p99 (edge-only vs escalated), macro-F1 of final
decisions, and the measured gateway service time (for the M/M/1 comparison)."""
import json
import os
import socket
import struct
import sys
import time

import numpy as np

BUNDLE = "/mnt/d/shield_run/outputs_v3/e6/e6_bundle.npz"
OUT = "/mnt/d/shield_run/outputs_v3/e6"
GW_PORT, CLOUD_PORT, REPLY_PORT = 9000, 9001, 9002


def student(B):
    W = [(B[f"W{i}"], B[f"b{i}"]) for i in range(int(B["n_layers"]))]

    def f(x):
        for i, (w, b) in enumerate(W):
            x = x @ w + b
            if i < len(W) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max()); p = e / e.sum()
        return int(p.argmax()), float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    return f


def run_cloud(cloud_ms, gw_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", CLOUD_PORT))
    r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data, _ = s.recvfrom(64)
        t_end = time.perf_counter() + cloud_ms / 1000
        while time.perf_counter() < t_end:          # busy-wait = deterministic service time
            pass
        r.sendto(data, (gw_ip, REPLY_PORT))


def run_gw(thr, cloud_ip, duration, logpath):
    B = np.load(BUNDLE); f = student(B); X = B["X"]; cloud_pred = B["cloud"]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", GW_PORT)); s.settimeout(0.001)
    rep = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rep.bind(("0.0.0.0", REPLY_PORT)); rep.setblocking(False)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rec, busy = [], 0.0
    t_stop = time.time() + duration + 3
    while time.time() < t_stop:
        try:
            data, _ = s.recvfrom(64)
            idx, t_sent = struct.unpack("!Id", data)
            t0 = time.perf_counter()
            pred, ent = f(X[idx])
            busy += time.perf_counter() - t0
            if ent > thr:
                out.sendto(struct.pack("!Id", idx, t_sent), (cloud_ip, CLOUD_PORT)); rec.append((idx, t_sent, None, 1, pred))
            else:
                rec.append((idx, t_sent, time.time(), 0, pred))
        except socket.timeout:
            pass
        while True:
            try:
                data, _ = rep.recvfrom(64)
            except BlockingIOError:
                break
            idx, t_sent = struct.unpack("!Id", data)
            rec.append((idx, t_sent, time.time(), 2, int(cloud_pred[idx])))
    json.dump({"rec": rec, "busy": busy}, open(logpath, "w"))


def run_client(ids, rate, duration):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gw = sys.argv[sys.argv.index("--gw") + 1]
    n = int(rate * duration); start = time.time()
    for k in range(n):
        target = start + k / rate
        while time.time() < target:
            pass
        s.sendto(struct.pack("!Id", int(ids[k % len(ids)]), time.time()), (gw, GW_PORT))


def analyse(logpath, R, b, duration, y, n_sent):
    L = json.load(open(logpath)); rec, busy = L["rec"], L["busy"]
    edge = {r[0]: r for r in rec if r[3] == 0}; esc = {r[0]: r for r in rec if r[3] == 1}; back = {r[0]: r for r in rec if r[3] == 2}
    lat_e = [1000 * (r[2] - r[1]) for r in edge.values()]
    lat_c = [1000 * (back[i][2] - back[i][1]) for i in esc if i in back]
    final = {**{i: r[4] for i, r in edge.items()}, **{i: back[i][4] for i in esc if i in back}}
    ids = np.array(list(final)); pred = np.array([final[i] for i in ids])
    from collections import Counter
    labels = sorted(set(y[ids].tolist()) | set(pred.tolist()))
    f1s = []
    for c in labels:
        tp = int(((pred == c) & (y[ids] == c)).sum()); fp = int(((pred == c) & (y[ids] != c)).sum()); fn = int(((pred != c) & (y[ids] == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    n_dec = len(final)
    return {"rate": R, "budget": b, "sent": n_sent, "decided": n_dec, "loss": 1 - n_dec / max(n_sent, 1),
            "escalated_frac": len(esc) / max(len(edge) + len(esc), 1),
            "cloud_returned_frac": len(lat_c) / max(len(esc), 1),
            "lat_edge_p50_ms": float(np.percentile(lat_e, 50)) if lat_e else None,
            "lat_edge_p99_ms": float(np.percentile(lat_e, 99)) if lat_e else None,
            "lat_esc_p50_ms": float(np.percentile(lat_c, 50)) if lat_c else None,
            "lat_esc_p99_ms": float(np.percentile(lat_c, 99)) if lat_c else None,
            "macro_f1": float(np.mean(f1s)) if f1s else None,
            "gw_service_us": 1e6 * busy / max(len(edge) + len(esc), 1)}


def main():
    role = sys.argv[1]
    if role == "cloud":
        run_cloud(float(sys.argv[2]), sys.argv[3]); return
    if role == "gw":
        run_gw(float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]); return
    if role == "client":
        B = np.load(BUNDLE); n = len(B["y"]); h, H = int(sys.argv[2]), int(sys.argv[3])
        run_client(np.arange(h, n, H), float(sys.argv[4]), float(sys.argv[5])); return
    # controller
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSBridge
    from mininet.log import setLogLevel
    setLogLevel("warning")
    cloud_ms = float(os.environ.get("CLOUD_MS", "2")); duration = float(os.environ.get("DUR", "15"))
    B = np.load(BUNDLE); y = B["y"]; thr = dict(zip(B["budgets"].tolist(), B["thresholds"].tolist()))
    H = int(os.environ.get("HOSTS", "8"))   # 8 senders: busy-waiting clients must not starve the gateway's CPU
    net = Mininet(switch=OVSBridge, link=TCLink, controller=None)
    s1 = net.addSwitch("s1")                 # one L2 segment; the WAN is the cloud's 10 ms / 100 Mbit link
    gw, cl = net.addHost("gw"), net.addHost("cloud")
    hs = [net.addHost(f"h{i}") for i in range(H)]
    for h in hs:
        net.addLink(h, s1, bw=100)
    net.addLink(gw, s1, bw=1000)
    net.addLink(cl, s1, bw=100, delay="10ms")
    net.start()
    gw_ip2 = gw.IP(); cl_ip = cl.IP()
    results = []
    me = os.path.abspath(__file__)
    for R in (1000, 5000, 10000, 20000):
        for b in (0, 5, 10, 20, 57):
            logp = f"/tmp/e6_gw_{R}_{b}.json"
            cp = cl.popen(["python3", me, "cloud", str(cloud_ms), gw_ip2])
            gp = gw.popen(["python3", me, "gw", str(thr[b]), cl_ip, str(duration), logp])
            time.sleep(1.0)
            cps = [h.popen(["python3", me, "client", str(i), str(H), str(R / H), str(duration), "--gw", gw.IP()]) for i, h in enumerate(hs)]
            for p in cps:
                p.wait()
            gp.wait(); cp.kill()
            r = analyse(logp, R, b, duration, y, int(R / H * duration) * H)
            r["cloud_ms"] = cloud_ms
            results.append(r); print(json.dumps(r), flush=True)
            json.dump(results, open(f"{OUT}/e6_results_cloud{int(cloud_ms)}ms.json", "w"), indent=1)
    net.stop()


if __name__ == "__main__":
    main()
