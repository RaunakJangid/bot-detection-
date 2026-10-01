import numpy as np
import torch
from torch import nn

from shield.xai.shap_analysis import explain, importance_from_phi, k_for_cumulative


def test_explain_true_class_linear_closed_form():
    """For a linear model with a constant background b, SHAP of class y is exactly W[y] * (x - b)."""
    torch.manual_seed(0)
    lin = nn.Linear(5, 4)
    b = np.full((20, 5), 0.3, dtype=np.float32)
    X = np.random.default_rng(0).normal(size=(30, 5)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 4, 30)
    phi = explain(lin, b, X, y, torch.device("cpu"), nsamples=10, chunk=7)
    expected = lin.weight.detach().numpy()[y] * (X - 0.3)
    np.testing.assert_allclose(phi, expected, atol=1e-5)


def test_importance_and_cutoff():
    phi = np.array([[1.0, 0.0, -3.0], [-1.0, 0.0, 1.0]])
    glob, per_class = importance_from_phi(phi, np.array([0, 1]), 2)
    assert glob.tolist() == [1.0, 0.0, 2.0]
    assert per_class[0].tolist() == [1.0, 0.0, 3.0]
    assert k_for_cumulative(glob, 0.6) == 1 and k_for_cumulative(glob, 0.95) == 2
