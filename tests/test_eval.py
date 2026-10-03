import numpy as np
import pandas as pd
import pytest

from shield.cross.transfer import convert, coral_map, dann_adapt
from shield.eval.stats import nemenyi_cd, ranks_and_friedman


def test_nemenyi_cd_matches_demsar():
    # k = 4 methods, N = 10 blocks: CD = 2.569 * sqrt(4 * 5 / 60)
    assert nemenyi_cd(4, 10) == pytest.approx(2.569 * np.sqrt(20 / 60))


def test_ranks_direction():
    t = pd.DataFrame({"a": [0.9, 0.8, 0.95], "b": [0.5, 0.6, 0.4], "c": [0.7, 0.7, 0.7]})
    ranks, p = ranks_and_friedman(t, higher_is_better=True)
    assert list(ranks.index) == ["a", "c", "b"] and ranks["a"] == 1.0
    assert p is not None and 0 <= p <= 1


def test_coral_matches_source_covariance():
    rng = np.random.default_rng(0)
    Xs = rng.multivariate_normal([0, 0, 0], [[1, 0.8, 0], [0.8, 1, 0], [0, 0, 2]], 20000)
    Xt = rng.multivariate_normal([3, -1, 0], [[4, 0, 0], [0, 0.5, 0], [0, 0, 1]], 20000)
    out = coral_map(Xt, Xs, Xt, eps=1e-6)
    np.testing.assert_allclose(np.cov(out, rowvar=False), np.cov(Xs, rowvar=False), atol=0.05)
    np.testing.assert_allclose(out.mean(0), Xs.mean(0), atol=0.05)


def test_dann_adapt_runs_and_keeps_shape():
    import torch
    from shield.models.student import StudentMLP
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Xs, ys = rng.normal(size=(300, 5)).astype(np.float32), rng.integers(0, 3, 300)
    Xt = (rng.normal(size=(300, 5)) + 1.5).astype(np.float32)
    m = StudentMLP(5, 3, [16, 8])
    out = dann_adapt(m, Xs, ys, Xt, {"steps": 20, "batch_size": 64, "lr": 1e-3}, torch.device("cpu"), seed=0)
    assert out(torch.as_tensor(Xt)).shape == (300, 3)
    assert any(not torch.equal(a, b) for a, b in zip(m.state_dict().values(), out.state_dict().values()))


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
