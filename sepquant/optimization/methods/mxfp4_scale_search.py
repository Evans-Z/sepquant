from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from sepquant.formats.fp_ops import quantize_e2m1, raw_e2m1_block_scale
from sepquant.models.hadamard import block_hadamard_last_dim, rotate_gram_block_hadamard
from sepquant.models.load import resolve_device
from sepquant.optimization.layerwise import LayerOptimizationContext, LayerOptimizationResult
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class MXFP4HessianScaleSearchOptimizer:
    activation_format: str = "none"
    exponent_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    objective: str = "block"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_hessian_scale_search"

    def optimize(self, context: LayerOptimizationContext) -> LayerOptimizationResult:
        if context.gram is None:
            return LayerOptimizationResult(
                layer_name=context.layer_name,
                spec=LayerQuantizationSpec(enabled=False),
                metrics={"reason": "missing_gram"},
            )

        compute_device = resolve_device(self.device)
        original_weight = context.module.weight.detach().to(device=compute_device, dtype=torch.float32)
        original_weight = _rotate_weight(original_weight, rotation=self.rotation)
        gram = _rotate_gram(context.gram, rotation=self.rotation).to(device=compute_device)
        optimized_weight, search_metrics = mxfp4_hessian_scale_search_weight(
            weight=original_weight,
            gram=gram,
            exponent_offsets=self.exponent_offsets,
            objective=self.objective,
            device=compute_device,
        )
        rel_mse = _relative_reconstruction_error(
            original_weight=original_weight,
            quantized_weight=optimized_weight,
            gram=gram,
        )
        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format="mxfp4",
                activation_format=self.activation_format,
                rotation=self.rotation,
                enabled=True,
            ),
            metrics={
                "method": self.name,
                "device": str(compute_device),
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **search_metrics,
            },
            optimized_weight=optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4ScaleSearchResult:
    quantized_weight: torch.Tensor
    selected_scales: torch.Tensor
    selected_offsets: torch.Tensor
    metrics: dict[str, Any]


def mxfp4_hessian_scale_search_weight(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    exponent_offsets: list[int],
    device: torch.device | str,
    objective: str = "block",
    block_size: int = 32,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Search representable MXFP4 power-of-two scale nudges under block-diagonal Hessian."""

    result = search_mxfp4_hessian_scales(
        weight=weight,
        gram=gram,
        exponent_offsets=exponent_offsets,
        objective=objective,
        device=device,
        block_size=block_size,
        eps=eps,
    )
    return result.quantized_weight, result.metrics


def search_mxfp4_hessian_scales(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    exponent_offsets: list[int],
    device: torch.device | str,
    objective: str = "block",
    block_size: int = 32,
    eps: float = 1e-12,
) -> MXFP4ScaleSearchResult:
    """Search MXFP4 block scales and return reusable row/block scale decisions."""

    if not exponent_offsets:
        raise ValueError("exponent_offsets must not be empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if objective not in {"identity", "diag", "block"}:
        raise ValueError(f"Unsupported MXFP4 scale objective: {objective}")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")

    compute_device = torch.device(device)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32)
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32)
    rows, columns = work_weight.shape

    if hessian.shape != (columns, columns):
        raise ValueError(
            f"Gram shape {tuple(hessian.shape)} does not match weight input dimension {columns}"
        )

    pad = _padding(columns, block_size)
    if pad:
        work_weight = F.pad(work_weight, (0, pad))
        hessian = F.pad(hessian, (0, pad, 0, pad))

    padded_columns = work_weight.shape[1]
    num_blocks = padded_columns // block_size
    quantized_weight = torch.empty_like(work_weight)
    selected_scales = torch.empty((rows, num_blocks), dtype=torch.float32, device=compute_device)
    selected_offsets = torch.empty((rows, num_blocks), dtype=torch.int64, device=compute_device)

    default_error = torch.zeros((), dtype=torch.float32, device=compute_device)
    searched_error = torch.zeros((), dtype=torch.float32, device=compute_device)

    for block_idx in range(num_blocks):
        begin = block_idx * block_size
        end = begin + block_size
        weight_block = work_weight[:, begin:end]
        hessian_block = hessian[begin:end, begin:end]
        base_scale = _mxfp4_block_scale(weight_block, eps=eps)

        best_scores: torch.Tensor | None = None
        best_quantized: torch.Tensor | None = None
        best_scales: torch.Tensor | None = None
        block_offsets = torch.empty(rows, dtype=torch.int64, device=compute_device)

        default_quantized = _quantize_with_offset(
            weight_block=weight_block,
            base_scale=base_scale,
            exponent_offset=0,
        )
        default_error = default_error + _block_scores(
            weight_block=weight_block,
            quantized_block=default_quantized,
            hessian_block=hessian_block,
            objective=objective,
        ).sum()

        for exponent_offset in exponent_offsets:
            candidate_scale = base_scale * (2.0**exponent_offset)
            quantized_block = _quantize_with_offset(
                weight_block=weight_block,
                base_scale=base_scale,
                exponent_offset=exponent_offset,
            )
            scores = _block_scores(
                weight_block=weight_block,
                quantized_block=quantized_block,
                hessian_block=hessian_block,
                objective=objective,
            )
            if best_scores is None or best_quantized is None:
                best_scores = scores
                best_quantized = quantized_block
                best_scales = candidate_scale.squeeze(1)
                block_offsets.fill_(exponent_offset)
                continue

            improved = scores < best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_quantized = torch.where(improved.unsqueeze(1), quantized_block, best_quantized)
            if best_scales is None:
                raise RuntimeError("Best MXFP4 scales were not initialized")
            best_scales = torch.where(improved, candidate_scale.squeeze(1), best_scales)
            block_offsets = torch.where(
                improved,
                torch.full_like(block_offsets, exponent_offset),
                block_offsets,
            )

        if best_scores is None or best_quantized is None or best_scales is None:
            raise RuntimeError("No MXFP4 scale candidates were evaluated")
        quantized_weight[:, begin:end] = best_quantized
        selected_scales[:, block_idx] = best_scales
        selected_offsets[:, block_idx] = block_offsets
        searched_error = searched_error + best_scores.sum()

    trimmed_weight = quantized_weight[:, :columns]
    offset_counts = _offset_counts(selected_offsets)
    metrics = {
        "block_size": block_size,
        "exponent_offsets": exponent_offsets,
        "scale_objective": objective,
        "selected_offset_counts": offset_counts,
        "block_diagonal_error": searched_error.item(),
        "default_block_diagonal_error": default_error.item(),
        "block_diagonal_error_ratio": (searched_error / default_error.clamp_min(1e-12)).item(),
    }
    return MXFP4ScaleSearchResult(
        quantized_weight=trimmed_weight,
        selected_scales=selected_scales,
        selected_offsets=selected_offsets,
        metrics=metrics,
    )


def _mxfp4_block_scale(blocks: torch.Tensor, *, eps: float) -> torch.Tensor:
    raw_scale = raw_e2m1_block_scale(blocks, eps)
    return torch.pow(2.0, _mxfp4_scale_exponent(raw_scale))


def _quantize_with_offset(
    *,
    weight_block: torch.Tensor,
    base_scale: torch.Tensor,
    exponent_offset: int,
) -> torch.Tensor:
    base_exponent = torch.log2(base_scale)
    scale = torch.pow(2.0, _clamp_e8m0_exponent(base_exponent + exponent_offset))
    return quantize_e2m1(weight_block / scale) * scale


def _mxfp4_scale_exponent(raw_scale: torch.Tensor) -> torch.Tensor:
    return _clamp_e8m0_exponent(torch.ceil(torch.log2(raw_scale)))


def _clamp_e8m0_exponent(exponent: torch.Tensor) -> torch.Tensor:
    return torch.clamp(exponent, min=-127, max=127)


def _rotate_weight(weight: torch.Tensor, *, rotation: str) -> torch.Tensor:
    if rotation == "none":
        return weight
    if rotation == "block_hadamard":
        return block_hadamard_last_dim(weight)
    raise ValueError(f"Unsupported rotation: {rotation}")


def _rotate_gram(gram: torch.Tensor, *, rotation: str) -> torch.Tensor:
    if rotation == "none":
        return gram
    if rotation == "block_hadamard":
        return rotate_gram_block_hadamard(gram)
    raise ValueError(f"Unsupported rotation: {rotation}")


def _block_scores(
    *,
    weight_block: torch.Tensor,
    quantized_block: torch.Tensor,
    hessian_block: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    error = weight_block - quantized_block
    if objective == "identity":
        return torch.sum(error.square(), dim=1)
    if objective == "diag":
        return torch.sum(error.square() * torch.diag(hessian_block).unsqueeze(0), dim=1)
    if objective != "block":
        raise ValueError(f"Unsupported MXFP4 scale objective: {objective}")
    return torch.sum((error @ hessian_block) * error, dim=1)


def _offset_counts(selected_offsets: torch.Tensor) -> dict[str, int]:
    unique_offsets, counts = torch.unique(selected_offsets.cpu(), return_counts=True)
    return {
        str(offset.item()): count.item()
        for offset, count in zip(unique_offsets, counts, strict=True)
    }


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
