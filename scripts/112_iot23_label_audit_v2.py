"""IoT-23 v2 vs v1 label comparison for the captures flagged by the audit (35-1, 43-1, 33-1, 3-1, 49-1, 39-1, 42-1, 9-1).

For each capture the v2 folder (mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/<capture>/) is listed, its
*zeek-conn-log.labeled file is streamed (not stored), SHA-256 of the exact bytes is recorded, and the same aggregates
and order-independent per-label digests as scripts/111 are computed. v1 and v2 are identical for a label group iff
count and digest match. Also records the labels.config text. Output: outputs_v5/iot23_audit/v2_compare.json."""
import hashlib
import importlib.util
import json
import re
import urllib.request
from pathlib import Path

B = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/"
CAPS = ["CTU-IoT-Malware-Capture-" + c for c in ("49-1", "35-1", "3-1", "39-1", "42-1", "9-1", "33-1", "43-1")]
OUT = Path("D:/shield_run/outputs_v5/iot23_audit")
s = importlib.util.spec_from_file_location("a111", Path(__file__).with_name("111_iot23_label_audit_v1.py"))
a111 = importlib.util.module_from_spec(s); s.loader.exec_module(a111)


def main():
    v1 = json.load(open(OUT / "v1_matrix.json")); res = {}
    if (OUT / "v2_compare.json").exists():
        res = json.load(open(OUT / "v2_compare.json"))
    for cap in CAPS:
        if cap in res:
            continue
        html = urllib.request.urlopen(B + cap + "/", timeout=120).read().decode("utf-8", "replace")
        files = re.findall(r'href="([^"?/][^"]*)"', html)
        lab = [f for f in files if f.endswith("conn-log.labeled")]; cfg = [f for f in files if f.endswith("labels.config")]
        if not lab:
            res[cap] = {"error": "no labelled file in v2 folder", "files": files}; continue
        sha = hashlib.sha256(); size = 0

        def lines():
            nonlocal size
            with urllib.request.urlopen(B + cap + "/" + lab[0], timeout=300) as r:
                for raw in r:
                    sha.update(raw); size += len(raw)
                    if raw[:1] != b"#":
                        yield raw.decode("utf-8", "replace")
        r = a111.audit_stream(lines())
        r.update({"file": lab[0], "bytes": size, "sha256": sha.hexdigest(),
                  "labels_config": urllib.request.urlopen(B + cap + "/" + cfg[0], timeout=120).read().decode("utf-8", "replace") if cfg else None})
        d1, d2 = v1[cap]["digest"], r["digest"]
        r["identical_label_groups"] = {k: (d1.get(k) == d2.get(k)) for k in sorted(set(d1) | set(d2))}
        r["all_labels_identical"] = all(r["identical_label_groups"].values()); r["v1_rows"] = v1[cap]["rows"]
        res[cap] = r
        print(f"{cap}: v1 rows {v1[cap]['rows']} v2 rows {r['rows']} identical {r['all_labels_identical']} "
              f"groups {r['identical_label_groups']} sha256 {r['sha256'][:16]}", flush=True)
        json.dump(res, open(OUT / "v2_compare.json", "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
