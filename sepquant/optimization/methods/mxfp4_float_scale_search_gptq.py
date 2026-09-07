from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from sepquant.formats.fp_ops import (
    quantize_e2m1,
    quantize_unsigned_fp,
    raw_e2m1_block_scale,
    unsigned_fp_code_from_value,
    unsigned_fp_value_from_code,
)
from sepquant.models.hadamard import block_hadamard_last_dim, rotate_gram_block_hadamard
from sepquant.models.load import resolve_device
from sepquant.optimization.layerwise import LayerOptimizationContext, LayerOptimizationResult
from sepquant.optimization.methods.hessian_regularization import (
    HessianRegularization,
    regularize_hessian_for_cholesky,
)
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class MXFP4FloatScaleSpec:
    """Per-block unsigned float scale parameters for MXFP4 variants."""

    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    block_size: int = 32

    @property
    def max_code(self) -> int:
        return (1 << (self.exponent_bits + self.mantissa_bits)) - 1


MXFP4_E4M3_SCALE_SPEC = MXFP4FloatScaleSpec(
    name="mxfp4_e4m3",
    exponent_bits=4,
    mantissa_bits=3,
    exponent_bias=7,
)
MXFP4_E5M3_SCALE_SPEC = MXFP4FloatScaleSpec(
    name="mxfp4_e5m3",
    exponent_bits=5,
    mantissa_bits=3,
    exponent_bias=15,
)


@dataclass(frozen=True)
class MXFP4FloatHessianScaleSearchOptimizer:
    scale_spec: MXFP4FloatScaleSpec
    activation_format: str = "none"
    scale_code_offsets: list[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    scale_objective: str = "block"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_float_hessian_scale_search"

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
        scale_search = search_mxfp4_float_hessian_scales(
            weight=original_weight,
            gram=gram,
            scale_spec=self.scale_spec,
            scale_code_offsets=self.scale_code_offsets,
            objective=self.scale_objective,
            device=compute_device,
        )
        rel_mse = _relative_reconstruction_error(
            original_weight=original_weight,
            quantized_weight=scale_search.quantized_weight,
            gram=gram,
        )
        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format=self.scale_spec.name,
                activation_format=self.activation_format,
                rotation=self.rotation,
                enabled=True,
            ),
            metrics={
                "method": self.name,
                "device": str(compute_device),
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **scale_search.metrics,
            },
            optimized_weight=scale_search.quantized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4FloatScaleSearchGPTQOptimizer:
    scale_spec: MXFP4FloatScaleSpec
    activation_format: str = "none"
    damp_percent: float = 0.01
    hessian_regularization: HessianRegularization = "scalar_damp"
    scale_code_offsets: list[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    scale_objective: str = "block"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_float_hessian_scale_search_gptq"

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
        scale_search = search_mxfp4_float_hessian_scales(
            weight=original_weight,
            gram=gram,
            scale_spec=self.scale_spec,
            scale_code_offsets=self.scale_code_offsets,
            objective=self.scale_objective,
            device=compute_device,
        )
        optimized_weight = gptq_quantize_weight_with_mxfp4_float_scales(
            weight=original_weight,
            gram=gram,
            selected_scales=scale_search.selected_scales,
            damp_percent=self.damp_percent,
            hessian_regularization=self.hessian_regularization,
            device=compute_device,
            block_size=self.scale_spec.block_size,
        )
        rel_mse = _relative_reconstruction_error(
            original_weight=original_weight,
            quantized_weight=optimized_weight,
            gram=gram,
        )
        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format=self.scale_spec.name,
                activation_format=self.activation_format,
                rotation=self.rotation,
                enabled=True,
            ),
            metrics={
                "method": self.name,
                "device": str(compute_device),
                "damp_percent": self.damp_percent,
                "hessian_regularization": self.hessian_regularization,
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **_prefix_metrics("scale_search", scale_search.metrics),
            },
            optimized_weight=optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4FloatDynamicScaleSearchGPTQOptimizer:
    scale_spec: MXFP4FloatScaleSpec
    activation_format: str = "none"
    damp_percent: float = 0.01
    hessian_regularization: HessianRegularization = "scalar_damp"
    scale_code_offsets: list[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    scale_objective: str = "identity"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_float_dynamic_scale_search_gptq"

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
        optimized_weight, dynamic_metrics = gptq_quantize_weight_with_dynamic_mxfp4_float_scale_search(
            weight=original_weight,
            gram=gram,
            scale_spec=self.scale_spec,
            scale_code_offsets=self.scale_code_offsets,
            scale_objective=self.scale_objective,
            damp_percent=self.damp_percent,
            hessian_regularization=self.hessian_regularization,
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
                weight_format=self.scale_spec.name,
                activation_format=self.activation_format,
                rotation=self.rotation,
                enabled=True,
            ),
            metrics={
                "method": self.name,
                "device": str(compute_device),
                "damp_percent": self.damp_percent,
                "hessian_regularization": self.hessian_regularization,
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **dynamic_metrics,
            },
            optimized_weight=optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4FloatScaleSearchResult:
    quantized_weight: torch.Tensor
    selected_scales: torch.Tensor
    selected_scale_code_offsets: torch.Tensor
    metrics: dict[str, Any]


def search_mxfp4_float_hessian_scales(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    scale_spec: MXFP4FloatScaleSpec,
    scale_code_offsets: list[int],
    device: torch.device | str,
    objective: str = "block",
    eps: float = 1e-12,
) -> MXFP4FloatScaleSearchResult:
    """Search MXFP4 unsigned float block-scale code nudges under a block-diagonal Hessian."""

    if not scale_code_offsets:
        raise ValueError("scale_code_offsets must not be empty")
    if scale_spec.block_size <= 0:
        raise ValueError("block_size must be positive")
    if objective not in {"identity", "diag", "block"}:
        raise ValueError(f"Unsupported MXFP4 float scale objective: {objective}")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")

    compute_device = torch.device(device)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32)
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32)
    rows, columns = work_weight.shape
    block_size = scale_spec.block_size

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
        base_scale_code = _mxfp4_float_base_scale_code(weight_block, scale_spec=scale_spec, eps=eps)

        best_scores: torch.Tensor | None = None
        best_quantized: torch.Tensor | None = None
        best_scales: torch.Tensor | None = None
        block_offsets = torch.empty(rows, dtype=torch.int64, device=compute_device)

        default_quantized, _default_scale = _quantize_with_scale_code_offset(
            weight_block=weight_block,
            base_scale_code=base_scale_code,
            scale_spec=scale_spec,
            scale_code_offset=0,
            eps=eps,
        )
        default_error = default_error + _block_scores(
            weight_block=weight_block,
            quantized_block=default_quantized,
            hessian_block=hessian_block,
            objective=objective,
        ).sum()

        for scale_code_offset in scale_code_offsets:
            quantized_block, candidate_scale = _quantize_with_scale_code_offset(
                weight_block=weight_block,
                base_scale_code=base_scale_code,
                scale_spec=scale_spec,
                scale_code_offset=scale_code_offset,
                eps=eps,
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
                block_offsets.fill_(scale_code_offset)
                continue

            improved = scores < best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_quantized = torch.where(improved.unsqueeze(1), quantized_block, best_quantized)
            if best_scales is None:
                raise RuntimeError("Best MXFP4 float scales were not initialized")
            best_scales = torch.where(improved, candidate_scale.squeeze(1), best_scales)
            block_offsets = torch.where(
                improved,
                torch.full_like(block_offsets, scale_code_offset),
                block_offsets,
            )

        if best_scores is None or best_quantized is None or best_scales is None:
            raise RuntimeError("No MXFP4 float scale candidates were evaluated")
        quantized_weight[:, begin:end] = best_quantized
        selected_scales[:, block_idx] = best_scales
        selected_offsets[:, block_idx] = block_offsets
        searched_error = searched_error + best_scores.sum()

    trimmed_weight = quantized_weight[:, :columns]
    metrics = {
        "block_size": block_size,
        "weight_format": scale_spec.name,
        "scale_exponent_bits": scale_spec.exponent_bits,
        "scale_mantissa_bits": scale_spec.mantissa_bits,
        "scale_exponent_bias": scale_spec.exponent_bias,
        "scale_code_offsets": scale_code_offsets,
        "scale_objective": objective,
        "selected_scale_code_offset_counts": _offset_counts(selected_offsets),
        "block_diagonal_error": searched_error.item(),
        "default_block_diagonal_error": default_error.item(),
        "block_diagonal_error_ratio": (searched_error / default_error.clamp_min(1e-12)).item(),
    }
    return MXFP4FloatScaleSearchResult(
        quantized_weight=trimmed_weight,
        selected_scales=selected_scales,
        selected_scale_code_offsets=selected_offsets,
        metrics=metrics,
    )


def gptq_quantize_weight_with_mxfp4_float_scales(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    selected_scales: torch.Tensor,
    damp_percent: float,
    hessian_regularization: HessianRegularization = "scalar_damp",
    device: torch.device | str,
    block_size: int = 32,
) -> torch.Tensor:
    """Run GPTQ compensation while quantizing columns with searched MXFP4 float scales."""

    compute_device = torch.device(device)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32).clone()
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32).clone()
    scales = selected_scales.detach().to(device=compute_device, dtype=torch.float32)
    rows, columns = work_weight.shape
    expected_blocks = (columns + block_size - 1) // block_size

    if hessian.shape != (columns, columns):
        raise ValueError(
            f"Gram shape {tuple(hessian.shape)} does not match weight input dimension {columns}"
        )
    if scales.shape != (rows, expected_blocks):
        raise ValueError(
            f"Scale shape {tuple(scales.shape)} does not match expected {(rows, expected_blocks)}"
        )

    quantized_weight = torch.zeros_like(work_weight)
    diag = torch.diag(hessian)
    dead = diag == 0
    if dead.any():
        hessian[dead, dead] = 1.0
        work_weight[:, dead] = 0.0

    hessian = regularize_hessian_for_cholesky(
        hessian,
        damp_percent=damp_percent,
        method=hessian_regularization,
    ).hessian
    hessian_inv = torch.linalg.cholesky(hessian)
    hessian_inv = torch.cholesky_inverse(hessian_inv)
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

    for column_idx in range(columns):
        column = work_weight[:, column_idx]
        block_idx = column_idx // block_size
        column_scale = scales[:, block_idx]
        diag_value = hessian_inv[column_idx, column_idx]
        quantized_column = quantize_e2m1(column / column_scale) * column_scale
        quantized_weight[:, column_idx] = quantized_column

        error = (column - quantized_column) / diag_value
        if column_idx + 1 < columns:
            compensation = error.unsqueeze(1) * hessian_inv[column_idx, column_idx + 1 :].unsqueeze(0)
            work_weight[:, column_idx + 1 :] -= compensation

    return quantized_weight


def gptq_quantize_weight_with_dynamic_mxfp4_float_scale_search(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    scale_spec: MXFP4FloatScaleSpec,
    scale_code_offsets: list[int],
    scale_objective: str,
    damp_percent: float,
    hessian_regularization: HessianRegularization = "scalar_damp",
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run block-wise GPTQ while searching MXFP4 float scales from each compensated block."""

    compute_device = torch.device(device)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32).clone()
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32).clone()
    rows, columns = work_weight.shape
    block_size = scale_spec.block_size

    if hessian.shape != (columns, columns):
        raise ValueError(
            f"Gram shape {tuple(hessian.shape)} does not match weight input dimension {columns}"
        )

    quantized_weight = torch.zeros_like(work_weight)
    diag = torch.diag(hessian)
    dead = diag == 0
    if dead.any():
        hessian[dead, dead] = 1.0
        work_weight[:, dead] = 0.0

    search_hessian = hessian.clone()
    regularized_hessian = regularize_hessian_for_cholesky(
        hessian,
        damp_percent=damp_percent,
        method=hessian_regularization,
    )
    hessian = regularized_hessian.hessian
    hessian_inv = torch.linalg.cholesky(hessian)
    hessian_inv = torch.cholesky_inverse(hessian_inv)
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

    selected_offset_counts: dict[str, int] = {}
    block_diagonal_error = torch.zeros((), dtype=torch.float32, device=compute_device)
    default_block_diagonal_error = torch.zeros((), dtype=torch.float32, device=compute_device)

    for begin in range(0, columns, block_size):
        end = min(begin + block_size, columns)
        scale_search = search_mxfp4_float_hessian_scales(
            weight=work_weight[:, begin:end],
            gram=search_hessian[begin:end, begin:end],
            scale_spec=MXFP4FloatScaleSpec(
                name=scale_spec.name,
                exponent_bits=scale_spec.exponent_bits,
                mantissa_bits=scale_spec.mantissa_bits,
                exponent_bias=scale_spec.exponent_bias,
                block_size=end - begin,
            ),
            scale_code_offsets=scale_code_offsets,
            objective=scale_objective,
            device=compute_device,
        )
        quantized_block = scale_search.quantized_weight
        quantized_weight[:, begin:end] = quantized_block

        _merge_counts(
            selected_offset_counts,
            scale_search.metrics["selected_scale_code_offset_counts"],
        )
        block_diagonal_error += torch.tensor(
            scale_search.metrics["block_diagonal_error"],
            dtype=torch.float32,
            device=compute_device,
        )
        default_block_diagonal_error += torch.tensor(
            scale_search.metrics["default_block_diagonal_error"],
            dtype=torch.float32,
            device=compute_device,
        )

        if end < columns:
            block_error = work_weight[:, begin:end] - quantized_block
            block_hessian_inv = hessian_inv[begin:end, begin:end]
            future_hessian_inv = hessian_inv[begin:end, end:]
            scaled_error = _right_solve_upper(block_error, block_hessian_inv)
            work_weight[:, end:] -= scaled_error @ future_hessian_inv

    return quantized_weight, {
        "block_size": block_size,
        "weight_format": scale_spec.name,
        "scale_exponent_bits": scale_spec.exponent_bits,
        "scale_mantissa_bits": scale_spec.mantissa_bits,
        "scale_exponent_bias": scale_spec.exponent_bias,
        "scale_code_offsets": scale_code_offsets,
        "scale_objective": scale_objective,
        "selected_scale_code_offset_counts": selected_offset_counts,
        "dynamic_block_diagonal_error": block_diagonal_error.item(),
        "dynamic_default_block_diagonal_error": default_block_diagonal_error.item(),
        "dynamic_block_diagonal_error_ratio": (
            block_diagonal_error / default_block_diagonal_error.clamp_min(1e-12)
        ).item(),
        "gptq_granularity": "block",
        **regularized_hessian.metrics,
    }


def _mxfp4_float_base_scale_code(
    weight_block: torch.Tensor,
    *,
    scale_spec: MXFP4FloatScaleSpec,
    eps: float,
) -> torch.Tensor:
    raw_scale = raw_e2m1_block_scale(weight_block, eps)
    base_block_scale = quantize_unsigned_fp(
        raw_scale.float(),
        exponent_bits=scale_spec.exponent_bits,
        mantissa_bits=scale_spec.mantissa_bits,
        exponent_bias=scale_spec.exponent_bias,
    )
    return unsigned_fp_code_from_value(
        base_block_scale,
        exponent_bits=scale_spec.exponent_bits,
        mantissa_bits=scale_spec.mantissa_bits,
        exponent_bias=scale_spec.exponent_bias,
    )


def _quantize_with_scale_code_offset(
    *,
    weight_block: torch.Tensor,
    base_scale_code: torch.Tensor,
    scale_spec: MXFP4FloatScaleSpec,
    scale_code_offset: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_code = torch.clamp(
        base_scale_code + scale_code_offset,
        min=0,
        max=scale_spec.max_code,
    )
    scale = torch.clamp(
        unsigned_fp_value_from_code(
            scale_code,
            exponent_bits=scale_spec.exponent_bits,
            mantissa_bits=scale_spec.mantissa_bits,
            exponent_bias=scale_spec.exponent_bias,
            dtype=weight_block.dtype,
        ),
        min=eps,
    )
    return quantize_e2m1(weight_block / scale) * scale, scale


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _right_solve_upper(values: torch.Tensor, upper_matrix: torch.Tensor) -> torch.Tensor:
    """Return X such that X @ upper_matrix = values."""

    return torch.linalg.solve_triangular(
        upper_matrix.T,
        values.T,
        upper=False,
    ).T


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
        raise ValueError(f"Unsupported MXFP4 float scale objective: {objective}")
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
