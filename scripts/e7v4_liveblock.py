#!/usr/bin/env python3
"""E7b v4: matched live ablation of the procedure in Mininet with Zeek (run as root inside WSL Ubuntu).

Same topology, detector, Zeek settings, feature code, commissioning window and decision rules as e7_liveblock.py
(rows 28am/28ap/28aq); only the experimental design is extended, so every configuration sees the same traffic plan.

Configurations (what may block a source after the 60 s benign commissioning window):
  lab           model at the lab threshold (>= 5 flagged flows in 10 s)
  commissioned  model at the commissioned threshold (99th percentile of the window's scores)
  rate_always   per-source rate rule only, always on: block if new flows in a 5 s Zeek-time window > max(10, 3 x Nmax)
  ks_rate       rate rule only, switched on while the KS warning is on (no model blocking)
  full          commissioned model + KS-gated rate rule (the guarded mode of e7_liveblock.py)
The KS warning is computed identically in every configuration (shadow where unused): every 50 post-window flows,
two-sample KS distance between the last 300 scores and the window's scores against the alpha = 0.001 critical value.
Every check is logged (time, D, critical value) so other thresholds can be evaluated offline.

Scenarios (traffic after the window; 6 benign hosts run the same synthetic background as e7 throughout):
  attacks   SYN flood, UDP flood, nmap scan, slow SYN probe (as e7)
  bursts    benign bursts only, no attack: firmware download (3 x 30 MB HTTP), reconnection storm (all 6 hosts,
            40 HTTP requests each within ~5 s), camera stream (UDP 1,000 B at 200 pkt/s, 20 s, one flow), discovery
            sweep (one host, UDP to 60 ports within ~3 s)
  distributed  4 low-rate SYN sources (~3.3 new flows/s each, below the per-source limit, 40 s) and a SYN flood with
            random spoofed sources (500 pkt/s, 20 s)
Output: outputs_v4/liveblock/<scenario>/<config>_r<k>.json (+ flows). Usage: e7v4_liveblock.py <scenario> <config> <k>"""
import collections
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink

DET = "/mnt/d/shield_run/outputs_v3/liveblock/detector.npz"
OUT = "/mnt/d/shield_run/outputs_v4/liveblock"
ZK = "/tmp/zk4"
WARM, COUNT, WINDOW = 60, 5, 10.0
SRV = "10.0.0.100"
BENIGN = [f"10.0.0.{i}" for i in range(1, 7)]
ATT = {"syn_flood": "10.0.0.51", "udp_flood": "10.0.0.52", "nmap_scan": "10.0.0.53", "slow_syn": "10.0.0.54",
       "low1": "10.0.0.61", "low2": "10.0.0.62", "low3": "10.0.0.63", "low4": "10.0.0.64", "spoof_flood": "10.0.0.65"}
MAX_BLOCKS = 300
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
BURST_PY = r'''
import socket, sys, time, urllib.request
kind, srv = sys.argv[1], sys.argv[2]
if kind == "firmware":
    for _ in range(3):
        urllib.request.urlopen(f"http://{srv}/fw.bin", timeout=30).read()
elif kind == "storm":
    for _ in range(40):
        try:
            urllib.request.urlopen(f"http://{srv}/", timeout=2).read()
        except Exception:
            pass
        time.sleep(0.12)
elif kind == "camera":
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.bind(("0.0.0.0", 40000)); end = time.time() + 20
    while time.time() < end:
        u.sendto(bytes(1000), (srv, 5353)); time.sleep(0.005)
elif kind == "discovery":
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for p in range(1900, 1960):
        u.sendto(b"M-SEARCH * HTTP/1.1\r\n", (srv, p)); time.sleep(0.05)
'''
REDEF = """redef tcp_inactivity_timeout = 3 sec;
redef udp_inactivity_timeout = 3 sec;
redef icmp_inactivity_timeout = 3 sec;
redef tcp_attempt_delay = 2 sec;
redef tcp_close_delay = 1 sec;
redef table_expire_interval = 1 sec;
"""
PLANS = {
    "attacks": [("syn_flood", 0, ATT["syn_flood"], "timeout -s INT 20 hping3 -S -p 80 -i u2000 {srv}"),
                ("udp_flood", 25, ATT["udp_flood"], "timeout -s INT 20 hping3 --udp -p 53 -i u2000 {srv}"),
                ("nmap_scan", 50, ATT["nmap_scan"], "timeout 30 nmap -sS -T4 -p 1-2000 {srv}"),
                ("slow_syn", 70, ATT["slow_syn"], "timeout -s INT 30 hping3 -S -p 22 -i u100000 {srv}")],
    "bursts": [("firmware", 0, BENIGN[0], "python3 /tmp/burst.py firmware {srv}"),
               ("storm", 25, "all", "python3 /tmp/burst.py storm {srv}"),
               ("camera", 50, BENIGN[1], "python3 /tmp/burst.py camera {srv}"),
               ("discovery", 80, BENIGN[2], "python3 /tmp/burst.py discovery {srv}")],
    "distributed": [("low1", 0, ATT["low1"], "timeout -s INT 40 hping3 -S -p 80 -i u300000 {srv}"),
                    ("low2", 0, ATT["low2"], "timeout -s INT 40 hping3 -S -p 80 -i u300000 {srv}"),
                    ("low3", 0, ATT["low3"], "timeout -s INT 40 hping3 -S -p 80 -i u300000 {srv}"),
                    ("low4", 0, ATT["low4"], "timeout -s INT 40 hping3 -S -p 80 -i u300000 {srv}"),
                    ("spoof_flood", 50, ATT["spoof_flood"], "timeout -s INT 20 hping3 -S --rand-source -p 80 -i u2000 {srv}")],
}


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


def run(scenario, config, rep):
    model_on = config in ("lab", "commissioned", "full")
    os.system("mn -c > /dev/null 2>&1; rm -rf " + ZK + "; mkdir -p " + ZK)
    open(f"{ZK}/redef.zeek", "w").write(REDEF)
    open("/tmp/benign.py", "w").write(BENIGN_PY); open("/tmp/sink.py", "w").write(SINK_PY)
    open("/tmp/burst.py", "w").write(BURST_PY)
    if not os.path.exists("/tmp/fw.bin") or os.path.getsize("/tmp/fw.bin") != 30 * 2 ** 20:
        os.system("head -c 31457280 /dev/urandom > /tmp/fw.bin")
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
          "flag_times": collections.defaultdict(collections.deque), "block_reason": {}, "warn_on": [], "ks_log": [],
          "src_ts": collections.defaultdict(collections.deque), "warm_rate_max": 0, "rate_thr": None,
          "recent": collections.deque(maxlen=300), "warning": False, "n_post": 0}
    stop = threading.Event()

    def block(src, why):
        if len(st["blocked"]) >= MAX_BLOCKS:
            return
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
            x = det.feats(r); sc = det.score(x)
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
            st["flows"].append({"t": rel, "src": src, "score": sc, "proto": r.get("proto"), "state": r.get("conn_state"),
                                "n5": len(sq)})
            if rel >= WARM:
                st["recent"].append(sc); st["n_post"] += 1
                if st["n_post"] % 50 == 0 and len(st["recent"]) >= 100:
                    n1, n2 = len(st["recent"]), len(st["warm_scores"])
                    crit = 1.949 * np.sqrt((n1 + n2) / (n1 * n2))          # two-sample KS, alpha = 0.001
                    d = ks(np.array(st["recent"]), np.array(st["warm_scores"]))
                    st["ks_log"].append((round(rel, 2), d, float(crit)))
                    w = d > crit
                    if w != st["warning"]:
                        st["warn_on"].append((round(rel, 2), bool(w)))
                    st["warning"] = w
            if src in st["blocked"] or rel < WARM:
                continue
            if model_on:
                thr = st["thr"]["lab" if config == "lab" else "commissioned"]
                if sc > thr:
                    q = st["flag_times"][src]; q.append(t_proc)
                    while q and t_proc - q[0] > WINDOW:
                        q.popleft()
                    if len(q) >= COUNT:
                        block(src, "model")
                        continue
            rate_on = config == "rate_always" or (config in ("ks_rate", "full") and st["warning"])
            if rate_on and st["rate_thr"] and len(sq) > st["rate_thr"]:
                block(src, "rate")

    th = threading.Thread(target=tail, daemon=True); th.start()
    time.sleep(max(0, WARM + 5 - (time.time() - T0)))
    starts = {}
    base = time.time()
    for name, off, who, cmd in PLANS[scenario]:
        time.sleep(max(0, base + off - time.time()))
        starts[name] = time.time() - T0
        for ip in (BENIGN if who == "all" else [who]):
            hosts[ip].cmd(cmd.format(srv=SRV) + f" > {ZK}/{name}_{ip}.out 2>&1 &")
    time.sleep(max(0, T0 + total + 8 - time.time()))
    stop.set()
    flows_dump = subprocess.run(["ovs-ofctl", "dump-flows", "s1"], capture_output=True, text=True).stdout
    dropped = {}
    for line in flows_dump.splitlines():
        if "nw_src=" in line and "drop" in line:
            ip = line.split("nw_src=")[1].split(",")[0].split(" ")[0]
            dropped[ip] = int(line.split("n_packets=")[1].split(",")[0])
    net.stop()
    res = {"scenario": scenario, "config": config, "rep": rep, "thresholds": st["thr"],
           "n_warm_flows": len(st["warm_scores"]), "rate_threshold_flows_per_5s": st["rate_thr"],
           "warm_rate_max": st["warm_rate_max"], "events": {}, "warning_changes": st["warn_on"], "ks_log": st["ks_log"],
           "block_reason": st["block_reason"], "blocked_at": st["blocked"], "n_blocked": len(st["blocked"])}
    cthr = st["thr"]["commissioned"]
    plan = PLANS[scenario]
    for j, (name, off, who, cmd) in enumerate(plan):
        ips = BENIGN if who == "all" else [who]
        t_end = starts[plan[j + 1][0]] if j + 1 < len(plan) and plan[j + 1][1] > off else starts[name] + 40
        sent = None
        for ip in ips:
            p = f"{ZK}/{name}_{ip}.out"
            out = open(p).read() if os.path.exists(p) else ""
            if "packets transmitted" in out:
                sent = (sent or 0) + int(out.split("packets transmitted")[0].split()[-1])
        win = [f for f in st["flows"] if starts[name] <= f["t"] < t_end]
        res["events"][name] = {"start_s": starts[name], "end_s": t_end, "sources": ips,
                               "blocked_at_s": {ip: st["blocked"][ip] for ip in ips if ip in st["blocked"]},
                               "blocked_before_event": [ip for ip in ips if st["blocked"].get(ip, 1e9) < starts[name]],
                               "time_to_block_s": min((st["blocked"][ip] - starts[name] for ip in ips
                                                       if st["blocked"].get(ip, -1) >= starts[name]), default=None),
                               "packets_sent": sent, "packets_dropped": sum(dropped.get(ip, 0) for ip in ips),
                               "flows_from_sources": sum(f["src"] in ips for f in st["flows"]),
                               "flows_flagged_commissioned": sum(f["src"] in ips and cthr is not None and f["score"] > cthr for f in st["flows"]),
                               "warning_during_event": any(starts[name] <= t < t_end and d > c for t, d, c in st["ks_log"]),
                               "flows_in_40s": len(win)}
    if scenario == "distributed":
        res["spoofed_sources_blocked"] = sum(1 for ip in st["blocked"] if ip not in BENIGN and ip not in ATT.values())
        res["spoofed_flows_seen"] = sum(1 for f in st["flows"] if f["src"] not in BENIGN and f["src"] not in ATT.values())
    bf = [f for f in st["flows"] if f["src"] in BENIGN and f["t"] >= WARM]
    res["benign"] = {"wrongly_blocked": {ip: t for ip, t in st["blocked"].items() if ip in BENIGN},
                     "n_flows_after_warm": len(bf),
                     "flagged_share_lab": float(np.mean([f["score"] > det.lab_thr for f in bf])) if bf else None,
                     "flagged_share_commissioned": float(np.mean([f["score"] > cthr for f in bf])) if bf and cthr else None,
                     "dropped_packets": sum(dropped.get(ip, 0) for ip in BENIGN)}
    res["warning_share_of_checks"] = float(np.mean([d > c for _, d, c in st["ks_log"]])) if st["ks_log"] else None
    d = f"{OUT}/{scenario}"
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(f"{d}/{config}_r{rep}.json", "w"), indent=1)
    json.dump(st["flows"], open(f"{d}/flows_{config}_r{rep}.json", "w"))
    print(scenario, config, rep, "blocked", st["block_reason"], "benign wrongly blocked", res["benign"]["wrongly_blocked"])


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
