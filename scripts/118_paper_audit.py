"""Mechanical audit of a built paper (PDF text + builder source). Checks the standing writing rules and internal
consistency; content and number checks against the evidence map are done separately.
Usage: python 118_paper_audit.py <paper.pdf> <builder.py> [abstract_word_limit]"""
import re
import subprocess
import sys


def pdftext(pdf):
    wsl = "/mnt/" + pdf[0].lower() + pdf[2:].replace("\\", "/")
    return subprocess.run(["wsl", "--", "pdftotext", wsl, "-"], capture_output=True, text=True, encoding="utf-8").stdout


def main():
    pdf, src = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    t = pdftext(pdf); k = max(t.rfind("References"), t.rfind("REFERENCES"))
    body, refs = (t[:k], t[k:]) if k > 0 else (t, "")
    flat = re.sub(r"-\n", "", body); flat = re.sub(r"\s+", " ", flat)
    issues = []
    for ch, name in (("—", "em dash"), (" – ", "spaced en dash")):
        n = flat.count(ch)
        if n:
            issues.append(f"{name}: {n}")
    fp = re.findall(r"\b(we|We|our|Our|us|Us|ours)\b", flat)
    if fp:
        issues.append(f"first person: {sorted(set(fp))} x{len(fp)}")
    times = re.findall(r"[^.]{0,40}\b\d+(?:\.\d+)?\s*(?:times|×|-fold|fold)\b[^.]{0,30}", flat)
    times += re.findall(r"[^.]{0,40}\btimes (?:more|less|faster|slower|higher|lower|larger|smaller)\b[^.]{0,30}", flat)
    if times:
        issues.append("N-times phrasing: " + " | ".join(x.strip() for x in times))
    m = re.search(r"Abstract\s*:?(.*?)Index Terms", flat)
    aw = len(m.group(1).split()) if m else None
    tabs = len(re.findall(r"\bTABLE [IVXL]+\b", body)); figs = len(re.findall(r"\bFig\. \d+\. ", body))
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"]
    reft = set(re.findall(r"\bTables? ([IVXL]+)\b", flat)); reff = set(re.findall(r"\bFigs?\. (\d+)", flat))
    for r in reft:
        if r not in roman[:tabs]:
            issues.append(f"Table {r} referenced but only {tabs} tables")
    for i in range(tabs):
        if roman[i] not in reft:
            issues.append(f"Table {roman[i]} never referenced in text")
    for f in reff:
        if int(f) > figs:
            issues.append(f"Fig. {f} referenced but only {figs} figures")
    for i in range(1, figs + 1):
        if str(i) not in reff:
            issues.append(f"Fig. {i} never referenced in text")
    nref = max([int(x) for x in re.findall(r"\[(\d+)\]", refs)] or [0])
    cites = [int(x) for grp in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", flat) for x in re.split(r"[,–-]", grp) if x.strip().isdigit()]
    over = sorted({x for x in cites if x > nref})
    if over:
        issues.append(f"citation numbers above reference count {nref}: {over}")
    srcs = open(src, encoding="utf-8").read()
    if "—" in srcs:
        issues.append("em dash in builder source")
    print(f"pages text ok | abstract words {aw} (limit {lim}) | tables {tabs} | figures {figs} | references {nref}")
    if aw and aw > lim:
        issues.append(f"abstract over limit: {aw} > {lim}")
    print("ISSUES:" if issues else "no mechanical issues")
    for i in issues:
        print(" -", i)


if __name__ == "__main__":
    main()
