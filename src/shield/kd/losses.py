"""Distillation losses.

L = alpha * CE(y, s) + beta * distill(s, t) + gamma * L_attr

distill = T^2 * KL(softmax(t/T) || softmax(s/T))                       ("kd", Hinton et al.)
        | dkd_alpha * TCKD + dkd_beta * NCKD                           ("dkd", Zhao et al. CVPR 2022)
  Decoupled KD separates the target-class binary distribution (TCKD) from the distribution over the
  non-target classes (NCKD), so the "dark knowledge" about wrong classes is not suppressed when the
  teacher is confident - the usual failure with a large teacher-student capacity gap.

L_attr = 1 - cos(w * A_s, w * A_t[S]) (e2KD, Parchami-Araghi et al. ECCV 2024): gradient x input of the
TEACHER'S PREDICTED class, so the student matches the teacher's reasoning (not just the ground truth),
re-weighted by normalised SHAP importance w over the student's features S.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)


def dkd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor, T: float,
        alpha: float = 1.0, beta: float = 8.0) -> torch.Tensor:
    """Decoupled KD = alpha * TCKD + beta * NCKD (both scaled by T^2)."""
    gt = F.one_hot(y, student_logits.shape[1]).to(student_logits.dtype)
    ps, pt = F.softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1)
    ps_t, pt_t = (ps * gt).sum(1, keepdim=True), (pt * gt).sum(1, keepdim=True)
    bs = torch.cat([ps_t, 1 - ps_t], 1).clamp_min(1e-8)
    bt = torch.cat([pt_t, 1 - pt_t], 1).clamp_min(1e-8)
    tckd = (bt * (bt.log() - bs.log())).sum(1).mean()
    # Non-target distribution: mask the target logit out before the softmax.
    mask = 1000.0 * gt
    nckd = F.kl_div(F.log_softmax(student_logits / T - mask, dim=1), F.softmax(teacher_logits / T - mask, dim=1),
                    reduction="batchmean")
    return (alpha * tckd + beta * nckd) * (T * T)


def grad_x_input(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient x input of logit y (per row). Returns (attributions, logits)."""
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
    """Batch layout: [x, y, teacher_logits?, teacher_attr?]. Teacher logits are present when beta > 0 or
    gamma > 0 (the attribution term needs the teacher's predicted class); attributions when gamma > 0."""

    def __init__(self, alpha: float, beta: float, gamma: float, T: float,
                 feature_weights: torch.Tensor | None = None, class_weight: torch.Tensor | None = None,
                 kd: str = "kd", dkd_alpha: float = 1.0, dkd_beta: float = 8.0):
        super().__init__()
        if kd not in ("kd", "dkd"):
            raise ValueError(f"Unknown distillation objective {kd!r}")
        self.alpha, self.beta, self.gamma, self.T = alpha, beta, gamma, T
        self.kd, self.dkd_alpha, self.dkd_beta = kd, dkd_alpha, dkd_beta
        self.feature_weights = feature_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    @property
    def needs_logits(self) -> bool:
        return self.beta > 0 or self.gamma > 0

    @property
    def needs_attr(self) -> bool:
        return self.gamma > 0

    def distill(self, s_logits: torch.Tensor, t_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kd == "dkd":
            return dkd(s_logits, t_logits, y, self.T, self.dkd_alpha, self.dkd_beta)
        return kd_kl(s_logits, t_logits, self.T)

    def forward(self, model: nn.Module, batch: list[torch.Tensor]) -> torch.Tensor:
        x, y, *extra = batch
        t_logits = extra.pop(0).float() if self.needs_logits else None
        t_attr = extra.pop(0).float() if self.needs_attr else None
        if self.needs_attr:
            a_s, s_logits = grad_x_input(model, x, t_logits.argmax(1), create_graph=True)
        else:
            s_logits = model(x)
        loss = self.alpha * self.ce(s_logits, y) if self.alpha > 0 else s_logits.new_zeros(())
        if self.beta > 0:
            loss = loss + self.beta * self.distill(s_logits, t_logits, y)
        if self.needs_attr:
            loss = loss + self.gamma * attribution_alignment(a_s, t_attr, self.feature_weights)
        return loss
