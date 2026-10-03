import numpy as np
import torch
from torch import nn

import pytest

from shield.kd.cache import compute_outputs
from shield.kd.losses import ShieldLoss, attribution_alignment, dkd, grad_x_input, kd_kl
from shield.kd.trainer import class_balanced_importance, resolve_spec
from shield.kd.tuning import choose_features, choose_objective, feature_grid, feature_name, objective_grid, \
    objective_name
from shield.models.student import StudentMLP, quantize_int8
from shield.models.teacher import FTTransformer, ResMLP


def test_kd_kl_zero_for_identical_logits():
    z = torch.randn(16, 5)
    assert kd_kl(z, z.clone(), 4.0).abs() < 1e-6
    assert kd_kl(z, torch.randn(16, 5), 4.0) > 0


def test_grad_x_input_linear_closed_form():
    lin = nn.Linear(4, 3, bias=False)
    x, y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
    a, _ = grad_x_input(lin, x, y, create_graph=False)
    torch.testing.assert_close(a, lin.weight[y] * x)


def test_alignment_zero_when_equal_and_scale_invariant():
    a = torch.randn(8, 6)
    w = torch.rand(6)
    assert attribution_alignment(a, a.clone(), w) < 1e-6
    assert attribution_alignment(a, 3 * a, w) < 1e-6
    assert attribution_alignment(a, -a, w) > 1.9


def test_shield_loss_student_equals_teacher():
    torch.manual_seed(0)
    m = StudentMLP(6, 4)
    x, y = torch.randn(32, 6), torch.randint(0, 4, (32,))
    pred = m(x).argmax(1)  # e2KD: attributions of the teacher's PREDICTED class
    a_t, t_logits = grad_x_input(m, x, pred, create_graph=False)
    for kd in ("kd", "dkd"):
        loss = ShieldLoss(alpha=0.0, beta=0.7, gamma=0.1, T=4.0, kd=kd)
        val = loss(m, [x, y, t_logits.detach(), a_t.detach()])
        assert val.abs() < 1e-4, kd  # distillation and attribution terms vanish when the student is the teacher


def test_dkd_decomposes_kd():
    """Zhao et al.: KD = TCKD + (1 - p_t^target) * NCKD, per sample."""
    torch.manual_seed(1)
    s, t, y = torch.randn(1, 5), torch.randn(1, 5), torch.tensor([2])
    T = 2.0
    tckd = dkd(s, t, y, T, alpha=1.0, beta=0.0)
    nckd = dkd(s, t, y, T, alpha=0.0, beta=1.0)
    p_target = torch.softmax(t / T, 1)[0, 2]
    torch.testing.assert_close(kd_kl(s, t, T), tckd + (1 - p_target) * nckd, rtol=1e-4, atol=1e-5)
    assert dkd(s, s.clone(), y, T) < 1e-6 and dkd(s, t, y, T) > 0


def test_dkd_loss_backward():
    torch.manual_seed(0)
    m = StudentMLP(6, 3)
    x, y = torch.randn(32, 6), torch.randint(0, 3, (32,))
    val = ShieldLoss(1.0, 1.0, 0.1, 2.0, kd="dkd", dkd_beta=8.0)(m, [x, y, torch.randn(32, 3), torch.randn(32, 6)])
    val.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_resolve_spec_precedence():
    cfg = {"loss": {"kd": "kd", "T": 4.0, "alpha": 0.3, "beta": 0.7, "gamma": 0.1, "dkd_alpha": 1.0, "dkd_beta": 8.0},
           "k": 16, "student": {"hidden": [64, 32]}}
    choice = {"objective": {"kd": "dkd", "T": 2.0, "alpha": 1.0, "beta": 1.0, "dkd_beta": 2.0, "teacher": "assistant"},
              "ranking": "class_balanced", "per_task": {"ciciot_family": {"k": 24, "hidden": [128, 64]}}}
    s = resolve_spec(cfg, {"select": "shap"}, "ciciot", "family", choice)
    assert (s["kd"], s["T"], s["k"], s["hidden"], s["ranking"], s["teacher"], s["gamma"]) == \
        ("dkd", 2.0, 24, [128, 64], "class_balanced", "assistant", 0.1)
    scratch = resolve_spec(cfg, {"select": "all", "alpha": 1.0, "beta": 0.0, "gamma": 0.0}, "ciciot", "family", choice)
    assert (scratch["alpha"], scratch["beta"], scratch["gamma"]) == (1.0, 0.0, 0.0)  # variant wins over tuning
    assert resolve_spec(cfg, {"select": "shap"}, "ciciot", "family", choice, k_override=8)["k"] == 8
    shared = resolve_spec(cfg, {"select": "shap"}, "xciciot", "binary", choice)
    assert shared["k"] == 16  # shared datasets keep the default budget
    assert resolve_spec(cfg, {"select": "shap"}, "ciciot", "family", None)["kd"] == "kd"


def test_class_balanced_importance_lifts_rare_class_features():
    # Feature 2 matters only for the rare class; global |SHAP| (dominated by class 0) ranks it last.
    info = {"per_class_importance": [[10.0, 5.0, 0.0], [0.0, 0.1, 1.0], [0.0, 0.0, 0.0]]}
    cb = class_balanced_importance(info)
    assert np.argsort(-cb)[0] == 2 or cb[2] > cb[1]
    assert np.isclose(cb.sum(), 1.0)


def test_tuning_choices():
    tcfg = {"objective": {"kd": {"T": [1.0, 4.0], "alpha": [0.3]}, "dkd": {"T": [2.0], "dkd_beta": [8.0]}},
            "features": {"ranking": ["global", "class_balanced"], "k": [8, 16], "hidden": [[64, 32]]}}
    grid = objective_grid(tcfg)
    assert len(grid) == 3 and grid[0]["beta"] == pytest.approx(0.7)
    scores = {objective_name(o): {"a": 0.5, "b": 0.5} for o in grid}
    scores[objective_name(grid[2])] = {"a": 0.6, "b": 0.7}
    assert choose_objective(scores, grid)["kd"] == "dkd"
    fgrid = feature_grid(tcfg)
    fs = {feature_name(c): {"t": 0.80 if c["ranking"] == "class_balanced" else 0.70} for c in fgrid}
    fs[feature_name({"ranking": "class_balanced", "k": 16, "hidden": [64, 32]})] = {"t": 0.802}
    params = {feature_name(c): {"t": c["k"] * 100} for c in fgrid}
    ranking, per_task = choose_features(fs, params, fgrid, ["t"], tolerance=0.005)
    # class-balanced wins; k=8 is within tolerance of the best (k=16) and cheaper
    assert ranking == "class_balanced" and per_task["t"] == {"k": 8, "hidden": [64, 32]}


def test_shield_loss_backward_reaches_all_params():
    torch.manual_seed(0)
    m = StudentMLP(6, 3)
    x, y = torch.randn(32, 6), torch.randint(0, 3, (32,))
    loss = ShieldLoss(0.3, 0.7, 0.5, 4.0, feature_weights=torch.rand(6))
    val = loss(m, [x, y, torch.randn(32, 3), torch.randn(32, 6)])
    val.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_teacher_cache_matches_live():
    torch.manual_seed(0)
    t = ResMLP(5, 3, width=16, depth=2).eval()
    X = np.random.default_rng(0).normal(size=(50, 5)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, 50)
    logits, attr = np.zeros((50, 3), np.float16), np.zeros((50, 5), np.float16)
    compute_outputs(t, X, torch.device("cpu"), logits, attr, batch=16)
    Xt = torch.as_tensor(X)
    live_a, live_l = grad_x_input(t, Xt, t(Xt).argmax(1), create_graph=False)  # predicted-class attributions
    np.testing.assert_allclose(logits, live_l.detach().numpy(), atol=2e-2, rtol=1e-2)
    np.testing.assert_allclose(attr, live_a.detach().numpy(), atol=2e-2, rtol=1e-2)


def test_models_forward_and_int8():
    x = torch.randn(4, 7)
    assert ResMLP(7, 3, width=16, depth=2).eval()(x).shape == (4, 3)
    assert FTTransformer(7, 3, d_token=16, n_layers=1, n_heads=2).eval()(x).shape == (4, 3)
    s = StudentMLP(7, 3).eval()
    q = quantize_int8(s)
    assert torch.allclose(s(x), q(x), atol=0.1)
