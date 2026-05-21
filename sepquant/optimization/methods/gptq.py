from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sepquant.formats import get_fp4_format
from sepquant.formats.fp_ops import quantize_e2m1
from sepquant.optimization.layerwise import LayerOptimizationContext, LayerOptimizationResult
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class GPTQOptimizer:
    weight_format: str
    activation_format: str = "none"
    damp_percent: float = 0.01
    device: str = "auto"
    name: str = "gptq"

    def optimize(self, context: LayerOptimizationContext) -> LayerOptimizationResult:
        if context.gram is None:
            return LayerOptimizationResult(
                layer_name=context.layer_name,
                spec=LayerQuantizationSpec(enabled=False),
                metrics={"reason": "missing_gram"},
            )

        compute_device = _resolve_compute_device(self.device)
        original_weight = context.module.weight.detach().to(device=compute_device, dtype=torch.float32)
        optimized_weight = gptq_quantize_weight(
            weight=original_weight,
            gram=context.gram,
            weight_format=self.weight_format,
            damp_percent=self.damp_percent,
            device=compute_device,
        )
        rel_mse = _relative_reconstruction_error(
            original_weight=original_weight,
            quantized_weight=optimized_weight,
            gram=context.gram,
        )
        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format=self.weight_format,
                activation_format=self.activation_format,
                enabled=True,
            ),
            metrics={
                "method": "gptq",
                "damp_percent": self.damp_percent,
                "device": str(compute_device),
                "weight_scale_granularity": "block",
                "relative_reconstruction_error": rel_mse,
            },
            optimized_weight=optimized_weight.cpu(),
        )


def gptq_quantize_weight(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    weight_format: str,
    damp_percent: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Column-wise GPTQ-style quantization with Hessian inverse compensation."""

    compute_device = _resolve_compute_device(device) if isinstance(device, str) else device
    quantizer = get_fp4_format(weight_format)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32).clone()
    quantized_weight = torch.zeros_like(work_weight)
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32).clone()
    columns = work_weight.shape[1]

    if hessian.shape != (columns, columns):
        raise ValueError(
            f"Gram shape {tuple(hessian.shape)} does not match weight input dimension {columns}"
        )

    diag = torch.diag(hessian)
    dead = diag == 0
    if dead.any():
        hessian[dead, dead] = 1.0
        work_weight[:, dead] = 0.0

    damp = damp_percent * torch.mean(torch.diag(hessian))
    hessian = hessian + torch.eye(columns, dtype=hessian.dtype, device=hessian.device) * damp
    hessian_inv = torch.linalg.cholesky(hessian)
    hessian_inv = torch.cholesky_inverse(hessian_inv)
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

    block_size = quantizer.block_size
    block_scales = None
    active_block_idx = -1
    for column_idx in range(columns):
        column = work_weight[:, column_idx]
        block_idx = column_idx // block_size
        if block_idx != active_block_idx:
            block_scales = _format_block_scales(work_weight, quantizer)
            active_block_idx = block_idx
        if block_scales is None:
            raise RuntimeError("Block scales were not initialized")

        diag_value = hessian_inv[column_idx, column_idx]
        column_scale = block_scales[:, block_idx]
        quantized_column = quantize_e2m1(column / column_scale) * column_scale
        quantized_weight[:, column_idx] = quantized_column

        error = (column - quantized_column) / diag_value
        if column_idx + 1 < columns:
            compensation = error.unsqueeze(1) * hessian_inv[column_idx, column_idx + 1 :].unsqueeze(0)
            work_weight[:, column_idx + 1 :] -= compensation

    return quantized_weight


def _format_block_scales(weight: torch.Tensor, quantizer) -> torch.Tensor:
    if not hasattr(quantizer, "_block_scale"):
        raise ValueError(f"Quantizer {quantizer.name} does not expose block-scale computation")

    rows, columns = weight.shape
    block_size = quantizer.block_size
    pad = _padding(columns, block_size)
    padded_weight = F.pad(weight, (0, pad)) if pad else weight
    blocks = padded_weight.reshape(rows, -1, block_size).reshape(-1, block_size)
    scales = quantizer._block_scale(blocks)
    return scales.reshape(rows, -1)


def _padding(width: int, block_size: int) -> int:
    remainder = width % block_size
    return 0 if remainder == 0 else block_size - remainder


def _relative_reconstruction_error(
    *,
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    gram: torch.Tensor,
) -> float:
    denominator = _quadratic_weight_error(original_weight, gram).clamp_min(1e-12)
    numerator = _quadratic_weight_error(original_weight - quantized_weight, gram)
    return (numerator / denominator).item()


def _quadratic_weight_error(weight_error: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    gram = gram.detach().to(device=weight_error.device, dtype=torch.float32)
    weight_error = weight_error.float()
    return torch.sum((weight_error @ gram) * weight_error)


def _resolve_compute_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested GPTQ device {device!r}, but CUDA is not available")
        return torch.device(device)
    if normalized in {"mps", "apple", "apple-silicon", "apple_silicon"}:
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(f"Requested GPTQ device {device!r}, but MPS is not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device)
