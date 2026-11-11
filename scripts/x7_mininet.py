#!/usr/bin/env python3
"""X7 (P2 v4): Mininet v2 emulation of the cascade under shifted replay, validation threshold vs online cap
(run as root inside WSL Ubuntu; numpy only).

Topology as e6_mininet.py: IoT hosts -> s1 (OVS) -> gateway; large-model host over a 10 ms / 100 Mbit link.
Replay: the shifted streams of scripts/90 in stream order (CICIoMT2024 6-class at 1,000 flows/s, 30 s; IoT-23 1-1
at 500 flows/s, the whole capture), host h sends flows h, h + H, ... so the global order is kept.
Gateway: real student outputs of the bundle, plus a busy-wait equal to the measured batch-1 student time; escalates
by (val) the validation threshold for the 10% budget or (cap) the running 90th percentile of the previous 300
entropies, refreshed every 50 flows (as scripts/85).
Large-model tier: one FIFO server, busy-wait service time sized by the paper's rule: 80% utilisation at the planned
share, s_t = 0.8 / (R x e_plan), e_plan = 10% (s10) or 13% (s13, the X9 sizing rule).
Per run: escalated share, share of escalations answered (3 s drain), latency p50/p99 of escalated decisions,
macro-F1 of final decisions (unanswered escalations fall back to the student decision).
Gateway and large-model sockets use 8 MB receive buffers (a first attempt on a loaded machine lost flows during
1 s scheduler stalls). Usage: python3 x7_mininet.py [reps]   Output: /mnt/d/shield_run/outputs_v5/p2/x7/runs.jsonl."""
import json
import os
import socket
import struct
import sys
import time
from collections import deque

import numpy as np

D = "/mnt/d/shield_run/outputs_v5/p2/x7"
GW_PORT, CLOUD_PORT, REPLY_PORT = 9000, 9001, 9002
SETTINGS = {"ciciomt6": 1000, "iot23_1-1": 500}


def bundle(name):
    return np.load(f"{D}/x7_{name}.npz")


def busy(sec):
    t = time.perf_counter() + sec
    while time.perf_counter() < t:
        pass


def rcvbuf(s):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)   # absorb short scheduler stalls without drops
    return s


def run_cloud(st, gw_ip):
    s = rcvbuf(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)); s.bind(("0.0.0.0", CLOUD_PORT))
    r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data, _ = s.recvfrom(64); busy(st); r.sendto(data, (gw_ip, REPLY_PORT))


def run_gw(name, mode, cloud_ip, duration, logpath):
    B = bundle(name); H, ps, pe, thr0, ss = B["H"], B["ps"], B["pe"], float(B["thr"]), float(B["student_s"])
    s = rcvbuf(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)); s.bind(("0.0.0.0", GW_PORT)); s.settimeout(0.001)
    rep = rcvbuf(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)); rep.bind(("0.0.0.0", REPLY_PORT)); rep.setblocking(False)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rec, win, thr, cnt = [], deque(maxlen=300), thr0, 0
    t_stop = time.time() + duration + 4
    while time.time() < t_stop:
        try:
            data, _ = s.recvfrom(64); idx, t_sent = struct.unpack("!Id", data)
            busy(ss); h = float(H[idx])
            if mode == "cap" and len(win) == 300:
                if cnt == 0:
                    thr = float(np.quantile(np.fromiter(win, float), 0.9)); cnt = 50
                cnt -= 1
            win.append(h)
            if h > thr:
                out.sendto(struct.pack("!Id", idx, t_sent), (cloud_ip, CLOUD_PORT)); rec.append((idx, t_sent, None, 1, int(ps[idx])))
            else:
                rec.append((idx, t_sent, time.time(), 0, int(ps[idx])))
        except socket.timeout:
            pass
        while True:
            try:
                data, _ = rep.recvfrom(64)
            except BlockingIOError:
                break
            idx, t_sent = struct.unpack("!Id", data); rec.append((idx, t_sent, time.time(), 2, int(pe[idx])))
    json.dump(rec, open(logpath, "w"))


def run_client(h, H, rate, n_total):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); gw = sys.argv[sys.argv.index("--gw") + 1]
    ids = np.arange(h, n_total, H); start = time.time()
    for k, i in enumerate(ids):
        target = start + k / rate
        while time.time() < target:
            pass
        s.sendto(struct.pack("!Id", int(i), time.time()), (gw, GW_PORT))


def analyse(name, rec, n_sent):
    y = bundle(name)["y"]
    edge = {r[0]: r for r in rec if r[3] == 0}; esc = {r[0]: r for r in rec if r[3] == 1}; back = {r[0]: r for r in rec if r[3] == 2}
    lat_c = [1000 * (back[i][2] - back[i][1]) for i in esc if i in back]
    final = {**{i: r[4] for i, r in edge.items()}, **{i: (back[i][4] if i in back else r[4]) for i, r in esc.items()}}
    ids = np.array(sorted(final)); pred = np.array([final[i] for i in ids]); yy = y[ids]
    f1 = [2 * ((pred == c) & (yy == c)).sum() / max(2 * ((pred == c) & (yy == c)).sum() + ((pred == c) & (yy != c)).sum()
          + ((pred != c) & (yy == c)).sum(), 1) for c in sorted(set(yy.tolist()) | set(pred.tolist()))]
    return {"sent": n_sent, "decided": len(final), "escalated_frac": len(esc) / max(len(final), 1),
            "answered_frac": len(lat_c) / max(len(esc), 1),
            "lat_esc_p50_ms": float(np.percentile(lat_c, 50)) if lat_c else None,
            "lat_esc_p99_ms": float(np.percentile(lat_c, 99)) if lat_c else None,
            "macro_f1": 100 * float(np.mean(f1))}


def main():
    role = sys.argv[1]
    if role == "cloud":
        run_cloud(float(sys.argv[2]), sys.argv[3]); return
    if role == "gw":
        run_gw(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]); return
    if role == "client":
        run_client(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])); return
    reps = int(role) if role.isdigit() else 3
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.net import Mininet
    from mininet.node import OVSBridge
    setLogLevel("warning")
    os.system("sysctl -qw net.core.rmem_max=16777216")
    NH = int(os.environ.get("HOSTS", "4"))
    net = Mininet(switch=OVSBridge, link=TCLink, controller=None)
    s1 = net.addSwitch("s1"); gw, cl = net.addHost("gw"), net.addHost("cloud"); hs = [net.addHost(f"h{i}") for i in range(NH)]
    for h in hs:
        net.addLink(h, s1, bw=100)
    net.addLink(gw, s1, bw=1000); net.addLink(cl, s1, bw=100, delay="10ms")
    net.start(); me = os.path.abspath(__file__)
    for node in (gw, cl):                      # the limit may be per network namespace
        node.cmd("sysctl -qw net.core.rmem_max=16777216")
    done = set()
    if os.path.exists(f"{D}/runs.jsonl"):
        done = {(r["setting"], r["sizing"], r["mode"], r["rep"]) for r in map(json.loads, open(f"{D}/runs.jsonl"))}
    for rep in range(1, reps + 1):
        for name, R in SETTINGS.items():
            n = min(len(bundle(name)["y"]), int(30 * R)); dur = n / R
            for sizing, ep in (("s10", 0.10), ("s13", 0.13)):
                st = 0.8 / (R * ep)
                for mode in ("val", "cap"):
                    if (name, sizing, mode, rep) in done:
                        continue
                    logp = f"/tmp/x7_{name}_{sizing}_{mode}_{rep}.json"
                    cp = cl.popen(["python3", me, "cloud", str(st), gw.IP()])
                    gp = gw.popen(["python3", me, "gw", name, mode, cl.IP(), str(dur), logp])
                    time.sleep(1.0)
                    cps = [h.popen(["python3", me, "client", str(i), str(NH), str(R / NH), str(n), "--gw", gw.IP()]) for i, h in enumerate(hs)]
                    for p in cps:
                        p.wait()
                    gp.wait(); cp.kill()
                    r = {"setting": name, "rate": R, "sizing": sizing, "service_ms": 1000 * st, "mode": mode, "rep": rep,
                         **analyse(name, json.load(open(logp)), n)}
                    print(json.dumps(r), flush=True)
                    with open(f"{D}/runs.jsonl", "a") as f:
                        f.write(json.dumps(r) + "\n")
    net.stop(); print("ALL_DONE")


if __name__ == "__main__":
    main()
