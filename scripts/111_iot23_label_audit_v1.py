"""IoT-23 label audit matrix from the raw v1 archive (every conn.log.labeled, all flows; data-quality evidence for P1/P2).

Per capture: rows, label x detailed-label counts; infected host = most frequent originator among Malicious flows;
aggregates by (originator is infected host, proto, conn_state, label, detailed); destination-port distribution
(ports >= 10000 bucketed) by (originator is infected host, proto, label); for the pre-specified candidate pattern
(originator = infected host, TCP, conn_state in {S0, REJ, RSTO, RSTOS0, SH}): flows and distinct destination IPs per label.
Order-independent label digest per capture and label: count and sum mod 2^64 of blake2b-64(uid + label + detailed), so a v2
file can be compared flow by flow without storing uids (scripts/112).
Output: outputs_v5/iot23_audit/v1_matrix.json."""
import collections
import hashlib
import json
import re
import tarfile
from pathlib import Path

ARCHIVE = r"D:\bot-datasets\iot23\iot_23_datasets_small.tar.gz"
OUT = Path("D:/shield_run/outputs_v5/iot23_audit")
SCAN = {"S0", "REJ", "RSTO", "RSTOS0", "SH"}
M64 = (1 << 64) - 1


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def audit_stream(lines):
    """lines: iterable of decoded data rows (no '#'); returns aggregates (infected host resolved at the end)."""
    agg = collections.Counter(); ports = collections.Counter(); hosts = collections.Counter(); labs = collections.Counter()
    dig = collections.defaultdict(lambda: [0, 0]); dst = collections.defaultdict(set); n = 0
    for s in lines:
        t = s.split()
        if len(t) < 23:
            continue
        n += 1; oh, rh, rp, pr, cs, lab, det = t[2], t[4], t[5], t[6], t[11], t[21], t[22]
        labs[(lab, det)] += 1
        if lab == "Malicious":
            hosts[oh] += 1
        p = rp if (rp.isdigit() and int(rp) < 10000) else ">=10000"
        agg[(oh, pr, cs, lab, det)] += 1; ports[(oh, pr, p, lab)] += 1
        if pr == "tcp" and cs in SCAN:
            dst[(oh, lab)].add(rh)
        d = dig[f"{lab}|{det}"]; d[0] += 1; d[1] = (d[1] + h64(t[1] + lab + det)) & M64
    bot = hosts.most_common(1)[0][0] if hosts else None
    role = lambda oh: "infected_host" if oh == bot else ("other_local" if oh.startswith("192.168.") else "external")
    A = collections.Counter(); Pp = collections.Counter()
    for (oh, pr, cs, lab, det), v in agg.items():
        A[(role(oh), pr, cs, lab, det)] += v
    for (oh, pr, p, lab), v in ports.items():
        Pp[(role(oh), pr, p, lab)] += v
    cand = {lab: {"flows": int(sum(v for (r, pr, cs, l, d), v in A.items() if r == "infected_host" and pr == "tcp" and cs in SCAN and l == lab)),
                  "distinct_dst": len(dst.get((bot, lab), ()))} for lab in ("Benign", "Malicious")}
    top = lambda C, k: [[*key, v] for key, v in C.most_common(k)]
    return {"rows": n, "infected_host": bot, "labels": [[*k, v] for k, v in labs.most_common()], "candidate_pattern": cand,
            "agg_top": top(A, 40), "ports_top": top(Pp, 40), "digest": {k: [v[0], str(v[1])] for k, v in dig.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    with tarfile.open(ARCHIVE, "r|gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith("conn.log.labeled")):
                continue
            cap = re.search(r"(CTU-[A-Za-z]+-?[A-Za-z]*-Capture-[\d-]+)", m.name).group(1)
            f = tf.extractfile(m)
            r = audit_stream(raw.decode("utf-8", "replace") for raw in f if raw[:1] != b"#")
            r["member"] = m.name; r["member_size"] = m.size; res[cap] = r
            c = r["candidate_pattern"]
            print(f"{cap:34s} rows {r['rows']:10d} bot {r['infected_host']} cand benign {c['Benign']['flows']} "
                  f"(dst {c['Benign']['distinct_dst']}) cand malicious {c['Malicious']['flows']} (dst {c['Malicious']['distinct_dst']})", flush=True)
            json.dump(res, open(OUT / "v1_matrix.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
