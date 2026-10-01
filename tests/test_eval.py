import numpy as np
import pandas as pd
import pytest

from shield.cross.transfer import convert
from shield.eval.stats import nemenyi_cd, ranks_and_friedman


def test_nemenyi_cd_matches_demsar():
    # k = 4 methods, N = 10 blocks: CD = 2.569 * sqrt(4 * 5 / 60)
    assert nemenyi_cd(4, 10) == pytest.approx(2.569 * np.sqrt(20 / 60))


def test_ranks_direction():
    t = pd.DataFrame({"a": [0.9, 0.8, 0.95], "b": [0.5, 0.6, 0.4], "c": [0.7, 0.7, 0.7]})
    ranks, p = ranks_and_friedman(t, higher_is_better=True)
    assert list(ranks.index) == ["a", "c", "b"] and ranks["a"] == 1.0
    assert p is not None and 0 <= p <= 1


def test_convert_restandardises_exactly():
    rng = np.random.default_rng(0)
    raw = rng.normal(3, 2, (50, 3))
    meta_a = {"scaler": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 1.0, 0.5]}}
    meta_b = {"scaler": {"mean": [0.0, 1.0], "std": [1.0, 4.0]}}
    Xa = (raw - meta_a["scaler"]["mean"]) / meta_a["scaler"]["std"]
    # feature order differs: dataset b has columns [f2, f0]
    out = convert(Xa, ["f0", "f1", "f2"], meta_a, ["f2", "f0"], meta_b)
    expected = (raw[:, [2, 0]] - np.array(meta_b["scaler"]["mean"])) / np.array(meta_b["scaler"]["std"])
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
