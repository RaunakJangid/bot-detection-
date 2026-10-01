"""Distillation losses.

L = alpha * CE(y, s) + beta * T^2 * KL(softmax(t/T) || softmax(s/T)) + gamma * L_attr

L_attr = 1 - cos(w * A_s, w * A_t[S]) aligns the student's gradient x input attribution for the
true class with the teacher's (restricted to the SHAP-selected subset S), with both re-weighted by
the normalised global SHAP importance w, so agreement on the most important features counts most.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)


def grad_x_input(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient x input of the true-class logit. Returns (attributions, logits)."""
    x = x.detach().requires_grad_(True)
    logits = model(x)
    selected = logits.gather(1, y[:, None]).sum()
    grad = torch.autograd.grad(selected, x, create_graph=create_graph)[0]
    return grad * x, logits


def attribution_alignment(a_student: torch.Tensor, a_teacher: torch.Tensor,
                          weights: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    if weights is not None:
        a_student, a_teacher = a_student * weights, a_teacher * weights
    cos = F.cosine_similarity(a_student, a_teacher.to(a_student.dtype), dim=1, eps=eps)
    return (1.0 - cos).mean()


class ShieldLoss(nn.Module):
    """Batch layout: [x, y, teacher_logits?, teacher_attr?] (optional tensors only when their weight > 0)."""

    def __init__(self, alpha: float, beta: float, gamma: float, T: float,
                 feature_weights: torch.Tensor | None = None, class_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.T = alpha, beta, gamma, T
        self.feature_weights = feature_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    @property
    def needs_logits(self) -> bool:
        return self.beta > 0

    @property
    def needs_attr(self) -> bool:
        return self.gamma > 0

    def forward(self, model: nn.Module, batch: list[torch.Tensor]) -> torch.Tensor:
        x, y, *extra = batch
        t_logits = extra.pop(0).float() if self.needs_logits else None
        t_attr = extra.pop(0).float() if self.needs_attr else None
        if self.needs_attr:
            a_s, s_logits = grad_x_input(model, x, y, create_graph=True)
        else:
            s_logits = model(x)
        loss = self.alpha * self.ce(s_logits, y) if self.alpha > 0 else s_logits.new_zeros(())
        if self.needs_logits:
            loss = loss + self.beta * kd_kl(s_logits, t_logits, self.T)
        if self.needs_attr:
            loss = loss + self.gamma * attribution_alignment(a_s, t_attr, self.feature_weights)
        return loss
