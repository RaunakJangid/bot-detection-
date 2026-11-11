#!/usr/bin/env python3
"""Cycle 1 pilot (K4'): sensor change vs traffic change on the P1 live testbed (run as root in WSL Ubuntu).

Borrowed unchanged from e7v5_liveblock.py: topology (server, 6 benign hosts), benign background, Zeek settings, the P1
live detector and its features, the burst plan. Added for this pilot only:
  - Zeek runs on a monitor host fed by an OVS mirror of the server port, so sensor-side changes do not touch traffic;
  - a control host (10.0.0.90) sends a fixed script of known flows: every 10 s an HTTP GET of a fixed 2 KB object, a
    100-byte UDP datagram and one ping; every 30 s a 5-SYN probe to port 22;
  - the condition is switched on at the end of the 60 s commissioning window:
      R0 nothing; S1 receive offload (GRO, LRO) enabled on the monitor interface; S2 1% loss on the mirror link (netem
      on the switch side of the monitor link); T1 the e7v5 benign burst plan (starts 5 s after the switch).
Logged per flow: time, source, detector score and the raw Zeek fields used by the features. Also the server-side
packet counters (traffic-untouched check) and the control flows sent.
Usage: python3 x8_sensor_pilot.py <R0|S1|S2|T1> <rep>   Output: runs/cycle_1/<cond>_r<rep>.json"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

S = importlib.util.spec_from_file_location("e7", "/mnt/c/Users/Prakash/shield/scripts/e7v5_liveblock.py")
e7 = importlib.util.module_from_spec(S); S.loader.exec_module(e7)
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSBridge

OUT = "/mnt/d/shield_run/papers/novelty/runs/cycle_1"
ZK = "/tmp/zk8"
WARM, POST = 60, 270
CTRL_IP = "10.0.0.90"
RAW = ["proto", "conn_state", "history", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
       "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "id.resp_p"]
CTRL_PY = r'''
import socket, subprocess, sys, time, urllib.request
srv, end = sys.argv[1], time.time() + float(sys.argv[2]); u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
k, log = 0, open(sys.argv[3], "w")
while time.time() < end:
    t = time.time()
    try:
        urllib.request.urlopen(f"http://{srv}/ctrl.txt", timeout=2).read(); log.write(f"{t} http\n")
    except Exception:
        log.write(f"{t} http_fail\n")
    u.sendto(bytes(100), (srv, 5353)); log.write(f"{t} udp\n")
    subprocess.run(["ping", "-c", "1", "-W", "1", srv], stdout=subprocess.DEVNULL); log.write(f"{t} icmp\n")
    if k % 3 == 0:
        subprocess.run(["hping3", "-S", "-p", "22", "-c", "5", "-i", "u200000", srv], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL); log.write(f"{t} syn5\n")
    log.flush(); k += 1
    time.sleep(max(0, t + 10 - time.time()))
'''


def counters(node, intf):
    out = node.cmd(f"cat /sys/class/net/{intf}/statistics/rx_packets /sys/class/net/{intf}/statistics/tx_packets").split()
    return [int(x) for x in out[:2]]


def run(cond, rep):
    os.system("mn -c > /dev/null 2>&1; rm -rf " + ZK + "; mkdir -p " + ZK)
    open(f"{ZK}/redef.zeek", "w").write(e7.REDEF)
    for path, src in (("/tmp/benign.py", e7.BENIGN_PY), ("/tmp/sink.py", e7.SINK_PY), ("/tmp/burst.py", e7.BURST_PY),
                      ("/tmp/ctrl.py", CTRL_PY)):
        open(path, "w").write(src)
    open("/tmp/ctrl.txt", "wb").write(b"c" * 2048)
    if not os.path.exists("/tmp/fw.bin") or os.path.getsize("/tmp/fw.bin") != 30 * 2 ** 20:
        os.system("head -c 31457280 /dev/urandom > /tmp/fw.bin")
    net = Mininet(switch=OVSBridge, controller=None, link=TCLink, autoSetMacs=True)
    s1 = net.addSwitch("s1")
    srv = net.addHost("srv", ip=e7.SRV + "/24")
    l_srv = net.addLink(srv, s1, bw=100)
    hosts = {}
    for ip in e7.BENIGN + [CTRL_IP]:
        h = net.addHost(f"h{ip.split('.')[-1]}", ip=ip + "/24")
        net.addLink(h, s1, bw=100)
        hosts[ip] = h
    mon = net.addHost("mon", ip="10.0.0.200/24")
    l_mon = net.addLink(mon, s1, cls=None) if False else net.addLink(mon, s1)
    net.start()
    p_srv, p_mon = l_srv.intf2.name, l_mon.intf2.name
    subprocess.run(f"ovs-vsctl -- --id=@p get port {p_srv} -- --id=@m get port {p_mon} -- --id=@mir create mirror "
                   f"name=m0 select-dst-port=@p select-src-port=@p output-port=@m -- set bridge s1 mirrors=@mir",
                   shell=True, capture_output=True)
    mon.cmd("ethtool -K mon-eth0 gro off lro off > /dev/null 2>&1; ip link set mon-eth0 promisc on")
    off0 = mon.cmd("ethtool -k mon-eth0 | grep -E 'generic-receive-offload|large-receive-offload'").strip()
    srv.cmd("cd /tmp && python3 -m http.server 80 > /dev/null 2>&1 &")
    srv.cmd("python3 /tmp/sink.py > /dev/null 2>&1 &")
    mon.cmd(f"cd {ZK} && /opt/zeek/bin/zeek -C -i mon-eth0 {ZK}/redef.zeek > {ZK}/zeek.out 2>&1 &")
    time.sleep(4)
    T0 = time.time(); total = WARM + POST
    for ip in e7.BENIGN:
        hosts[ip].cmd(f"python3 /tmp/benign.py {e7.SRV} {total} > /dev/null 2>&1 &")
    hosts[CTRL_IP].cmd(f"python3 /tmp/ctrl.py {e7.SRV} {total} {ZK}/ctrl_sent.log > /dev/null 2>&1 &")
    det = e7.Detector()
    flows, stop = [], threading.Event()

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
            if src == e7.SRV:
                continue
            sc = det.score(det.feats(r))
            flows.append({"t": time.time() - T0, "src": src, "score": sc, **{k: r.get(k) for k in RAW}})

    th = threading.Thread(target=tail, daemon=True); th.start()
    time.sleep(max(0, WARM - (time.time() - T0)))
    c_switch = counters(srv, "srv-eth0"); t_switch = time.time() - T0
    if cond == "S1":
        mon.cmd("ethtool -K mon-eth0 gro on lro on > /dev/null 2>&1")
    elif cond == "S2":
        subprocess.run(f"tc qdisc replace dev {p_mon} root netem loss 1%", shell=True)
    elif cond == "T1":
        base = time.time() + 5
        for name, off, who, cmd in e7.PLANS["bursts"]:
            time.sleep(max(0, base + off - time.time()))
            for ip in (e7.BENIGN if who == "all" else [who]):
                hosts[ip].cmd(cmd.format(srv=e7.SRV) + f" > {ZK}/{name}_{ip}.out 2>&1 &")
    off1 = mon.cmd("ethtool -k mon-eth0 | grep -E 'generic-receive-offload|large-receive-offload'").strip()
    time.sleep(max(0, T0 + total + 6 - time.time()))
    c_end = counters(srv, "srv-eth0")
    stop.set()
    zeek_err = open(f"{ZK}/zeek.out").read()[-500:] if os.path.exists(f"{ZK}/zeek.out") else ""
    ctrl_sent = open(f"{ZK}/ctrl_sent.log").read().split("\n") if os.path.exists(f"{ZK}/ctrl_sent.log") else []
    net.stop()
    os.makedirs(OUT, exist_ok=True)
    res = {"cond": cond, "rep": rep, "warm_s": WARM, "post_s": POST, "t_switch": t_switch,
           "srv_counters_switch": c_switch, "srv_counters_end": c_end, "offload_before": off0, "offload_after": off1,
           "ctrl_sent": [l for l in ctrl_sent if l], "zeek_tail": zeek_err, "n_flows": len(flows), "flows": flows}
    json.dump(res, open(f"{OUT}/{cond}_r{rep}.json", "w"))
    print(cond, rep, "flows", len(flows), "ctrl flows", sum(f["src"] == CTRL_IP for f in flows),
          "offload", off0.replace("\n", " "), "->", off1.replace("\n", " "), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
