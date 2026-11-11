"""Process-level snapshot of the machine before a timing run (reproducibility of the timing environment).
Writes JSON: time, total CPU % (2 s), per-core %, RAM, and every process with its CPU % over 2 s, RSS and command line.
Usage: python 115_proc_snapshot.py <out.json>"""
import json
import sys
import time

import psutil

procs = list(psutil.process_iter(["pid", "name", "cmdline"]))
for p in procs:
    try:
        p.cpu_percent(None)
    except psutil.Error:
        pass
total = psutil.cpu_percent(2.0, percpu=True)
rows = []
for p in procs:
    try:
        rows.append({"pid": p.pid, "name": p.info["name"], "cpu_percent": p.cpu_percent(None),
                     "rss_mb": round(p.memory_info().rss / 2**20, 1), "cmdline": " ".join(p.info["cmdline"] or [])[:300]})
    except psutil.Error:
        pass
rows.sort(key=lambda r: -r["cpu_percent"])
json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_total_percent": sum(total) / len(total), "cpu_per_core": total,
           "ram_used_percent": psutil.virtual_memory().percent, "n_processes": len(rows), "processes": rows},
          open(sys.argv[1], "w"), indent=1)
print(f"snapshot: cpu {sum(total) / len(total):.1f}%, top {[(r['name'], r['cpu_percent']) for r in rows[:5]]}")
