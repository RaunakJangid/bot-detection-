"""v3: check that ciciot_full is directly comparable with the strict ciciot dataset: identical feature list
and identical validation/test labels row for row (every task), and no train vector equal to an evaluation
vector. Exits non-zero on any mismatch."""

import sys

import numpy as np

from shield.cli import parser
from shield.utils.config import Paths
from shield.utils.io import load_json


def main():
    args = parser(__doc__, datasets=False).parse_args()
    paths = Paths(args.smoke)
    a, b = paths.processed("ciciot"), paths.processed("ciciot_full")
    ma, mb = load_json(a / "meta.json"), load_json(b / "meta.json")
    ok = ma["features"] == mb["features"] and ma["tasks"] == mb["tasks"]
    for task in ma["tasks"]:
        for s in ("validation", "test"):
            same = np.array_equal(np.load(a / f"y_{task}_{s}.npy"), np.load(b / f"y_{task}_{s}.npy"))
            print(f"{task} {s}: labels identical = {same}")
            ok &= same
    print("train rows: strict", ma["n_rows"]["train"], "-> full", mb["n_rows"]["train"])
    print("dedup:", mb["dedup"])
    print("COMPARABLE" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
