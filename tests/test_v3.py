import numpy as np
import pytest
import torch

from shield.eval.cascade import cascade_curve, cascade_predict, threshold_for_budget, confidence
from shield.eval.openset import FlyHash, exceedance, union_score
from shield.eval.posthoc import apply_bias, fit_decision_rule
from shield.models.quant import IntMLP, calibrate
from shield.models.student import StudentMLP


def _toy(n=3000, c=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(c, n, p=[0.7, 0.2, 0.07, 0.03])
    z = rng.normal(size=(n, c)).astype(np.float32)
    z[np.arange(n), y] += 1.5
    return z, y


def test_decision_rule_never_lowers_validation_f1():
    z, y = _toy()
    prior = np.bincount(y, minlength=4) / len(y)
    r = fit_decision_rule(z, y, prior, torch.device("cpu"))
    assert r["val_macro_f1_la"] >= r["val_macro_f1_raw"] - 1e-9
    assert r["val_macro_f1_bias"] >= r["val_macro_f1_la"] - 1e-9
    assert apply_bias(z, r["bias"]).shape == z.shape


def test_int_mlp_matches_float_model():
    torch.manual_seed(0)
    model = StudentMLP(8, 3, [16, 8]).eval()
    X = np.random.default_rng(0).normal(size=(5000, 8)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.as_tensor(X)).numpy()
    for bits, agree in ((8, 0.97), (16, 0.995)):   # an untrained model has many near-ties
        q = IntMLP(model, calibrate(model, X, 100.0), act_bits=bits)
        assert (q.forward(X).argmax(1) == ref.argmax(1)).mean() >= agree
        assert q.quantize_input(X).dtype == (np.int8 if bits == 8 else np.int16)
    assert IntMLP(model, calibrate(model, X)).size_bytes() < 4 * sum(p.numel() for p in model.parameters())


def test_cascade_budget_endpoints():
    zs, y = _toy(seed=1)
    ze = np.eye(4, dtype=np.float32)[y] * 10   # a perfect expert
    t = threshold_for_budget(confidence(zs), 0.2)
    _, esc = cascade_predict(zs, ze, t)
    assert abs(esc.mean() - 0.2) < 0.01
    curve = cascade_curve(zs, ze, y, zs, ze, y, 4, [0.0, 0.2, 1.0])
    assert curve[0]["test_escalated"] == 0 and curve[-1]["test_escalated"] == 1
    assert curve[-1]["test_macro_f1"] == pytest.approx(1.0)
    assert curve[0]["test_macro_f1"] <= curve[1]["test_macro_f1"] <= curve[2]["test_macro_f1"]


def test_flyhash_flags_shifted_inputs():
    # Benign traffic varies only in 3 of 10 features (structured, like flow features); novel inputs vary in
    # the others. (FlyHash tags encode the direction of the centred input, so an isotropic benign cloud
    # would leave nothing novel to detect.)
    rng = np.random.default_rng(0)
    benign = np.zeros((20000, 10), np.float32)
    benign[:, :3] = rng.normal(size=(20000, 3))
    novel = np.zeros((2000, 10), np.float32)
    novel[:, 5:] = rng.normal(size=(2000, 5))
    fly = FlyHash(10, m=500, wta=16).fit(benign)
    assert fly.score(novel).mean() > fly.score(benign[:2000]).mean() + 0.05
    ref = np.arange(100.0)
    assert exceedance(ref, np.array([99.0]))[0] == pytest.approx(0.01)
    assert union_score(ref, ref, np.array([0.0]), np.array([99.0]))[0] == pytest.approx(0.99)
