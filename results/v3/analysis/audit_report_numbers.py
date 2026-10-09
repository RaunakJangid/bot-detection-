"""Audit: every number in the hand-written text of REPORT_TEMPLATE.md must appear in the generated tables or
facts.json (after rounding), or be listed for manual checking."""
import json
import re
from pathlib import Path

import pandas as pd

V3 = Path(__file__).resolve().parents[1]
text = re.sub(r"\{\{\w+\}\}", "", (V3 / "REPORT_TEMPLATE.md").read_text(encoding="utf-8"))
vals = []
for f in (V3 / "tables").glob("*.csv"):
    for v in pd.read_csv(f).to_numpy().ravel():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals += [float(x) for x in re.findall(r"-?\d+\.?\d*", str(v))]


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
    elif isinstance(o, (int, float)):
        yield float(o)


facts = list(walk(json.load(open(V3 / "data" / "facts.json"))))
pool = vals + facts + [100 * f for f in facts]


def found(x: float, dec: int) -> bool:
    tol = 0.5 * 10 ** -dec + 1e-9
    return any(abs(x - v) <= tol for v in pool) or any(abs(x - abs(v)) <= tol for v in pool)


missing = []
for m in re.finditer(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:e-?\d+)?", text):
    s = m.group(0).replace(",", "")
    x = float(s)
    dec = len(s.split(".")[1].split("e")[0]) if "." in s else 0
    if "e" in s or not found(x, dec):
        line = text[:m.start()].count("\n") + 1
        missing.append((line, s, text.splitlines()[line - 1].strip()[:110]))
print(f"{len(missing)} numbers not found automatically (check by hand):")
for line, s, ctx in missing:
    print(f"  L{line}: {s:>10} | {ctx}")
