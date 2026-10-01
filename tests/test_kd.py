import numpy as np
import torch
from torch import nn

from shield.kd.cache import compute_teacher_outputs
from shield.kd.losses import ShieldLoss, attribution_alignment, grad_x_input, kd_kl
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
    with torch.enable_grad():
        a_t, t_logits = grad_x_input(m, x, y, create_graph=False)
    loss = ShieldLoss(alpha=0.0, beta=0.7, gamma=0.1, T=4.0)
    val = loss(m, [x, y, t_logits.detach(), a_t.detach()])
    assert val.abs() < 1e-5  # KD and attribution terms vanish when the student is the teacher


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
    compute_teacher_outputs(t, X, y, torch.device("cpu"), logits, attr, batch=16)
    live_a, live_l = grad_x_input(t, torch.as_tensor(X), torch.as_tensor(y), create_graph=False)
    np.testing.assert_allclose(logits, live_l.detach().numpy(), atol=2e-2, rtol=1e-2)
    np.testing.assert_allclose(attr, live_a.detach().numpy(), atol=2e-2, rtol=1e-2)


def test_models_forward_and_int8():
    x = torch.randn(4, 7)
    assert ResMLP(7, 3, width=16, depth=2).eval()(x).shape == (4, 3)
    assert FTTransformer(7, 3, d_token=16, n_layers=1, n_heads=2).eval()(x).shape == (4, 3)
    s = StudentMLP(7, 3).eval()
    q = quantize_int8(s)
    assert torch.allclose(s(x), q(x), atol=0.1)
