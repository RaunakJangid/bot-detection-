#!/usr/bin/env python3
"""E7b: live detect-and-block in Mininet with Zeek (run as root inside WSL Ubuntu; numpy + mininet only).

Topology: one OVS switch; server srv (HTTP + UDP sink), 6 benign IoT hosts, 4 attackers. Zeek 8 listens on the
server's interface (short inactivity timeouts so conn.log records appear within seconds). A detector thread tails
conn.log, computes the IoT-23 Zeek features (no service one-hots), scores each flow with the exported small MLP
(scripts/61_export_liveblock.py), and blocks a source IP with an OpenFlow drop rule (ovs-ofctl) once it has >= 5
flagged flows within 10 s (the same count rule idea as Sapkota et al. 2025).
Timeline: 0-60 s benign only (commissioning window), then attacks on the server: SYN flood (500 pkt/s, 20 s), UDP
flood (500 pkt/s, 20 s), nmap SYN scan (ports 1-2000), slow SYN probe (10 pkt/s, 30 s). Benign traffic runs
throughout. Runs: mode `lab` (threshold = IoT-23 validation 99th benign percentile), mode `commissioned`
(threshold = 99th percentile of the scores seen in the 60 s benign window; label-free given a clean window), and mode
`guarded` = commissioned + the label-free score-shift warning of row 28aj: every 50 flows the KS distance between
the last 300 scores and the commissioning scores is compared with the alpha = 0.001 two-sample critical value; while
the warning is on, the model is not trusted alone and a per-source rate fallback (also learned in the benign window)
blocks any source whose new-flow count in a 5 s Zeek-time window exceeds max(10, 3 x the largest benign per-source
count seen while commissioning). In each run the other thresholds are evaluated in shadow (flags only).
Per run: per attacker time-to-block, packets sent vs dropped by the rule; benign hosts wrongly blocked; share of
benign flows flagged; Zeek-to-decision lag and per-flow scoring time."""
import collections
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink

DET = "/mnt/d/shield_run/outputs_v3/liveblock/detector.npz"
OUT = "/mnt/d/shield_run/outputs_v3/liveblock"
ZK = "/tmp/zk"
WARM, COUNT, WINDOW = 60, 5, 10.0
SRV = "10.0.0.100"
BENIGN = [f"10.0.0.{i}" for i in range(1, 7)]
ATT = {"syn_flood": "10.0.0.51", "udp_flood": "10.0.0.52", "nmap_scan": "10.0.0.53", "slow_syn": "10.0.0.54"}
PROTO = ["tcp", "udp", "icmp"]
STATES = ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
HIST = list("ShADadFfRrTtWwIQ^")
NUM = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]

BENIGN_PY = r'''
import random, socket, subprocess, sys, time, urllib.request
srv = sys.argv[1]; end = time.time() + float(sys.argv[2]); u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while time.time() < end:
    r = random.random()
    try:
        if r < 0.6:
            urllib.request.urlopen(f"http://{srv}/", timeout=2).read()
        elif r < 0.85:
            u.sendto(bytes(random.randint(20, 200)), (srv, 5353))
        else:
            subprocess.run(["ping", "-c", "1", "-W", "1", srv], stdout=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(random.uniform(0.3, 2.0))
'''
SINK_PY = r'''
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", 5353))
while True:
    s.recvfrom(2048)
'''
REDEF = """redef tcp_inactivity_timeout = 3 sec;
redef udp_inactivity_timeout = 3 sec;
redef icmp_inactivity_timeout = 3 sec;
redef tcp_attempt_delay = 2 sec;
redef tcp_close_delay = 1 sec;
redef table_expire_interval = 1 sec;
"""


class Detector:
    def __init__(self):
        B = np.load(DET)
        self.W = [(B[f"W{i}"], B[f"b{i}"]) for i in range(int(B["n_layers"]))]
        self.mu, self.sd, self.lab_thr = B["mu"], B["sd"], float(B["thr"])
        self.names = [str(n) for n in B["names"]]

    def feats(self, r):
        v = {}
        for c in NUM:
            try:
                v[c] = float(r[c])
            except (ValueError, KeyError):
                v[c] = np.nan
        v["bpp_orig"] = v["orig_ip_bytes"] / v["orig_pkts"] if v["orig_pkts"] else np.nan
        v["bpp_resp"] = v["resp_ip_bytes"] / v["resp_pkts"] if v["resp_pkts"] else np.nan
        for p in PROTO:
            v[f"proto_{p}"] = float(r.get("proto") == p)
        for s in STATES:
            v[f"st_{s}"] = float(r.get("conn_state") == s)
        h = r.get("history", "-")
        for ch in HIST:
            v[f"h_{ch}"] = float(h.count(ch))
        v["h_len"] = float(len(h))
        x = np.array([v[n] for n in self.names], dtype=np.float64)
        x = np.nan_to_num(np.sign(x) * np.log1p(np.abs(x)))
        return ((x - self.mu) / self.sd).astype(np.float32)

    def score(self, x):
        for i, (w, b) in enumerate(self.W):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max())
        return float(e[1] / e.sum())


def run(mode):
    os.system("mn -c > /dev/null 2>&1; rm -rf " + ZK + "; mkdir -p " + ZK)
    open(f"{ZK}/redef.zeek", "w").write(REDEF)
    open("/tmp/benign.py", "w").write(BENIGN_PY); open("/tmp/sink.py", "w").write(SINK_PY)
    net = Mininet(switch=OVSBridge, controller=None, link=TCLink, autoSetMacs=True)
    s1 = net.addSwitch("s1")
    srv = net.addHost("srv", ip=SRV + "/24")
    net.addLink(srv, s1, bw=100)
    hosts = {}
    for ip in BENIGN + list(ATT.values()):
        h = net.addHost(f"h{ip.split('.')[-1]}", ip=ip + "/24")
        net.addLink(h, s1, bw=100)
        hosts[ip] = h
    net.start()
    srv.cmd("cd /tmp && python3 -m http.server 80 > /dev/null 2>&1 &")
    srv.cmd("python3 /tmp/sink.py > /dev/null 2>&1 &")
    srv.cmd(f"cd {ZK} && /opt/zeek/bin/zeek -C -i srv-eth0 {ZK}/redef.zeek > {ZK}/zeek.out 2>&1 &")
    time.sleep(4)
    T0 = time.time()
    total = WARM + 120
    for ip in BENIGN:
        hosts[ip].cmd(f"python3 /tmp/benign.py {SRV} {total} > /dev/null 2>&1 &")
    det = Detector()
    st = {"thr": {"lab": det.lab_thr, "commissioned": None}, "blocked": {}, "warm_scores": [], "flows": [],
          "flag_times": collections.defaultdict(collections.deque), "block_reason": {}, "warn_on": [],
          "src_ts": collections.defaultdict(collections.deque), "warm_rate_max": 0, "rate_thr": None,
          "recent": collections.deque(maxlen=300), "warning": False, "n_post": 0}
    stop = threading.Event()

    def block(src, why):
        subprocess.run(["ovs-ofctl", "add-flow", "s1", f"priority=100,ip,nw_src={src},actions=drop"])
        st["blocked"][src] = time.time() - T0
        st["block_reason"][src] = why

    def ks(a, b):
        a, b = np.sort(a), np.sort(b)
        v = np.concatenate([a, b])
        return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))

    def tail():
        path = f"{ZK}/conn.log"
        while not os.path.exists(path) and not stop.is_set():
            time.sleep(0.2)
        fields, f = None, open(path)
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.05); continue
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]; continue
            if line.startswith("#") or fields is None:
                continue
            r = dict(zip(fields, line.rstrip("\n").split("\t")))
            src = r.get("id.orig_h")
            if src == SRV:
                continue
            t_proc = time.time()
            x = det.feats(r); t1 = time.perf_counter(); sc = det.score(x); dt = time.perf_counter() - t1
            try:
                lag = t_proc - (float(r["ts"]) + (float(r["duration"]) if r["duration"] not in ("-", "") else 0.0))
            except ValueError:
                lag = float("nan")
            rel = t_proc - T0
            try:
                zts = float(r["ts"])
            except ValueError:
                zts = t_proc
            sq = st["src_ts"][src]; sq.append(zts)
            while sq and zts - sq[0] > 5.0:
                sq.popleft()
            if rel < WARM:
                st["warm_scores"].append(sc)
                st["warm_rate_max"] = max(st["warm_rate_max"], len(sq))
            elif st["thr"]["commissioned"] is None:
                ws = np.array(st["warm_scores"])
                st["thr"]["commissioned"] = float(np.quantile(ws, 0.99)) if len(ws) >= 50 else det.lab_thr
                st["rate_thr"] = max(10, 3 * st["warm_rate_max"])
            st["flows"].append({"t": rel, "src": src, "score": sc, "lag": lag, "score_ms": 1000 * dt,
                                "proto": r.get("proto"), "state": r.get("conn_state")})
            thr = st["thr"]["commissioned" if mode == "guarded" else mode]
            if rel >= WARM:
                st["recent"].append(sc); st["n_post"] += 1
                if mode == "guarded" and st["n_post"] % 50 == 0 and len(st["recent"]) >= 100:
                    n1, n2 = len(st["recent"]), len(st["warm_scores"])
                    crit = 1.949 * np.sqrt((n1 + n2) / (n1 * n2))          # two-sample KS, alpha = 0.001
                    w = ks(np.array(st["recent"]), np.array(st["warm_scores"])) > crit
                    if w != st["warning"]:
                        st["warn_on"].append((round(rel, 2), bool(w)))
                    st["warning"] = w
            if src in st["blocked"] or rel < WARM:
                continue
            if thr is not None and sc > thr:
                q = st["flag_times"][src]; q.append(t_proc)
                while q and t_proc - q[0] > WINDOW:
                    q.popleft()
                if len(q) >= COUNT:
                    block(src, "model")
                    continue
            if mode == "guarded" and st["warning"] and st["rate_thr"] and len(sq) > st["rate_thr"]:
                block(src, "rate_fallback")

    th = threading.Thread(target=tail, daemon=True); th.start()
    time.sleep(max(0, WARM + 5 - (time.time() - T0)))
    starts = {}
    plan = [("syn_flood", 0, "timeout -s INT 20 hping3 -S -p 80 -i u2000 {srv}"),
            ("udp_flood", 25, "timeout -s INT 20 hping3 --udp -p 53 -i u2000 {srv}"),
            ("nmap_scan", 50, "timeout 30 nmap -sS -T4 -p 1-2000 {srv}"),
            ("slow_syn", 70, "timeout -s INT 30 hping3 -S -p 22 -i u100000 {srv}")]
    base = time.time()
    for name, off, cmd in plan:
        time.sleep(max(0, base + off - time.time()))
        starts[name] = time.time() - T0
        hosts[ATT[name]].cmd(cmd.format(srv=SRV) + f" > {ZK}/{name}.out 2>&1 &")
    time.sleep(max(0, T0 + total + 8 - time.time()))
    stop.set()
    flows_dump = subprocess.run(["ovs-ofctl", "dump-flows", "s1"], capture_output=True, text=True).stdout
    dropped = {}
    for line in flows_dump.splitlines():
        if "nw_src=" in line and "drop" in line:
            ip = line.split("nw_src=")[1].split(",")[0].split(" ")[0]
            dropped[ip] = int(line.split("n_packets=")[1].split(",")[0])
    net.stop()
    res = {"mode": mode, "thresholds": st["thr"], "n_warm_flows": len(st["warm_scores"]), "attacks": {}, "benign": {},
           "rate_threshold_flows_per_5s": st["rate_thr"], "warm_rate_max": st["warm_rate_max"],
           "warning_changes": st["warn_on"], "block_reason": st["block_reason"]}
    for name, ip in ATT.items():
        out = open(f"{ZK}/{name}.out").read() if os.path.exists(f"{ZK}/{name}.out") else ""
        sent = None
        if "packets transmitted" in out:
            sent = int(out.split("packets transmitted")[0].split()[-1])
        fl = [f for f in st["flows"] if f["src"] == ip]
        thr = st["thr"]["commissioned" if mode == "guarded" else mode]
        res["attacks"][name] = {"start_s": starts.get(name), "blocked_at_s": st["blocked"].get(ip),
                                "time_to_block_s": (st["blocked"][ip] - starts[name]) if ip in st["blocked"] else None,
                                "packets_sent": sent, "packets_dropped": dropped.get(ip, 0),
                                "flows_seen": len(fl), "flows_flagged": sum(f["score"] > thr for f in fl)}
    bf = [f for f in st["flows"] if f["src"] in BENIGN and f["t"] >= WARM]
    for m in ("lab", "commissioned"):
        thr = st["thr"][m]
        res["benign"][f"flagged_share_{m}"] = float(np.mean([f["score"] > thr for f in bf])) if bf and thr is not None else None
    res["benign"]["wrongly_blocked"] = {ip: t for ip, t in st["blocked"].items() if ip in BENIGN}
    res["benign"]["n_flows_after_warm"] = len(bf)
    lags = [f["lag"] for f in st["flows"] if f["lag"] == f["lag"]]
    res["latency"] = {"zeek_to_decision_s_p50": float(np.median(lags)) if lags else None,
                      "zeek_to_decision_s_p95": float(np.quantile(lags, 0.95)) if lags else None,
                      "score_ms_p50": float(np.median([f["score_ms"] for f in st["flows"]])) if st["flows"] else None}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/result_{mode}.json", "w"), indent=1)
    json.dump(st["flows"], open(f"{OUT}/flows_{mode}.json", "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    for mode in (sys.argv[1:] or ["lab", "commissioned"]):
        run(mode)
