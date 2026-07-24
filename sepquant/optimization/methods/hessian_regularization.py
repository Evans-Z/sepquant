from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

HessianRegularization = Literal["scalar_damp", "diagonal_dominance"]
HESSIAN_REGULARIZATION_CHOICES = ("scalar_damp", "diagonal_dominance")


@dataclass(frozen=True)
class RegularizedHessian:
    hessian: torch.Tensor
    metrics: dict[str, float | str]


def regularize_hessian_for_cholesky(
    hessian: torch.Tensor,
    *,
    damp_percent: float,
    method: HessianRegularization,
    eps: float = 1e-8,
) -> RegularizedHessian:
    """Apply diagonal regularization before Cholesky-based GPTQ compensation."""

    if method == "scalar_damp":
        damp = damp_percent * torch.mean(torch.diag(hessian))
        regularized = hessian + torch.eye(
            hessian.shape[0],
            dtype=hessian.dtype,
            device=hessian.device,
        ) * damp
        damp_value = damp.item()
        return RegularizedHessian(
            hessian=regularized,
            metrics={
                "hessian_regularization": method,
                "hessian_damp": damp_value,
                "hessian_diag_offset_mean": damp_value,
                "hessian_diag_offset_max": damp_value,
            },
        )

    if method == "diagonal_dominance":
        diag = torch.diag(hessian)
        offdiag_abs_sum = torch.sum(torch.abs(hessian), dim=1) - torch.abs(diag)
        offset = torch.clamp(offdiag_abs_sum - diag + eps, min=0.0)
        regularized = hessian + torch.diag(offset)
        return RegularizedHessian(
            hessian=regularized,
            metrics={
                "hessian_regularization": method,
                "hessian_damp": 0.0,
                "hessian_diag_offset_mean": offset.mean().item(),
                "hessian_diag_offset_max": offset.max().item(),
            },
        )

    raise ValueError(f"Unsupported Hessian regularization method: {method}")
