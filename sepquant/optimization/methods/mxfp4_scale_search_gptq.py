from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from sepquant.formats import get_fp4_format
from sepquant.formats.fp_ops import quantize_e2m1
from sepquant.models.hadamard import block_hadamard_last_dim, rotate_gram_block_hadamard
from sepquant.models.load import resolve_device
from sepquant.optimization.layerwise import LayerOptimizationContext, LayerOptimizationResult
from sepquant.optimization.methods.mxfp4_scale_search import search_mxfp4_hessian_scales
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class MXFP4ScaleSearchGPTQOptimizer:
    activation_format: str = "none"
    damp_percent: float = 0.01
    exponent_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    scale_objective: str = "block"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_hessian_scale_search_gptq"

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
        scale_search = search_mxfp4_hessian_scales(
            weight=original_weight,
            gram=gram,
            exponent_offsets=self.exponent_offsets,
            objective=self.scale_objective,
            device=compute_device,
        )
        optimized_weight = gptq_quantize_weight_with_mxfp4_scales(
            weight=original_weight,
            gram=gram,
            selected_scales=scale_search.selected_scales,
            damp_percent=self.damp_percent,
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
                "damp_percent": self.damp_percent,
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **_prefix_metrics("scale_search", scale_search.metrics),
            },
            optimized_weight=optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4DynamicScaleSearchGPTQOptimizer:
    activation_format: str = "none"
    damp_percent: float = 0.01
    exponent_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    scale_objective: str = "identity"
    rotation: str = "none"
    device: str = "auto"
    name: str = "mxfp4_dynamic_scale_search_gptq"

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
        optimized_weight, dynamic_metrics = gptq_quantize_weight_with_dynamic_mxfp4_scale_search(
            weight=original_weight,
            gram=gram,
            exponent_offsets=self.exponent_offsets,
            scale_objective=self.scale_objective,
            damp_percent=self.damp_percent,
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
                "damp_percent": self.damp_percent,
                "rotation": self.rotation,
                "relative_reconstruction_error": rel_mse,
                **dynamic_metrics,
            },
            optimized_weight=optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class MXFP4RotationSelectGPTQOptimizer:
    activation_format: str = "none"
    damp_percent: float = 0.01
    exponent_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    scale_objective: str = "identity"
    candidate_rotation: str = "block_hadamard"
    dynamic_scale_search: bool = True
    selection_margin: float = 0.0
    device: str = "auto"
    name: str = "mxfp4_dynamic_scale_search_gptq_rotation_select"

    def optimize(self, context: LayerOptimizationContext) -> LayerOptimizationResult:
        inputs = context.inputs
        gram = context.gram
        if inputs is None:
            return LayerOptimizationResult(
                layer_name=context.layer_name,
                spec=LayerQuantizationSpec(enabled=False),
                metrics={"reason": "missing_inputs"},
            )
        if gram is None:
            gram = inputs.float().t().matmul(inputs.float())

        compute_device = resolve_device(self.device)
        original_weight = context.module.weight.detach().to(device=compute_device, dtype=torch.float32)
        bias = (
            context.module.bias.detach().to(device=compute_device, dtype=torch.float32)
            if context.module.bias is not None
            else None
        )
        candidate_inputs = inputs.detach().to(device=compute_device, dtype=torch.float32)
        gram = gram.to(device=compute_device, dtype=torch.float32)

        no_rotation = _run_mxfp4_gptq_candidate(
            weight=original_weight,
            bias=bias,
            inputs=candidate_inputs,
            gram=gram,
            activation_format=self.activation_format,
            rotation="none",
            exponent_offsets=self.exponent_offsets,
            scale_objective=self.scale_objective,
            damp_percent=self.damp_percent,
            dynamic_scale_search=self.dynamic_scale_search,
            device=compute_device,
        )
        rotated = _run_mxfp4_gptq_candidate(
            weight=original_weight,
            bias=bias,
            inputs=candidate_inputs,
            gram=gram,
            activation_format=self.activation_format,
            rotation=self.candidate_rotation,
            exponent_offsets=self.exponent_offsets,
            scale_objective=self.scale_objective,
            damp_percent=self.damp_percent,
            dynamic_scale_search=self.dynamic_scale_search,
            device=compute_device,
        )

        threshold = no_rotation.output_relative_error * (1.0 - self.selection_margin)
        selected = rotated if rotated.output_relative_error < threshold else no_rotation
        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format="mxfp4",
                activation_format=self.activation_format,
                rotation=selected.rotation,
                enabled=True,
            ),
            metrics={
                "method": self.name,
                "device": str(compute_device),
                "damp_percent": self.damp_percent,
                "rotation": selected.rotation,
                "candidate_rotation": self.candidate_rotation,
                "selection_margin": self.selection_margin,
                "selection_metric": "output_relative_error",
                "no_rotation_output_relative_error": no_rotation.output_relative_error,
                "rotated_output_relative_error": rotated.output_relative_error,
                "no_rotation_weight_relative_reconstruction_error": no_rotation.weight_relative_error,
                "rotated_weight_relative_reconstruction_error": rotated.weight_relative_error,
                **_prefix_metrics(f"selected_{selected.rotation}", selected.metrics),
            },
            optimized_weight=selected.optimized_weight.cpu(),
        )


@dataclass(frozen=True)
class _MXFP4GPTQCandidate:
    rotation: str
    optimized_weight: torch.Tensor
    output_relative_error: float
    weight_relative_error: float
    metrics: dict[str, Any]


def gptq_quantize_weight_with_mxfp4_scales(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    selected_scales: torch.Tensor,
    damp_percent: float,
    device: torch.device | str,
    block_size: int = 32,
) -> torch.Tensor:
    """Run GPTQ compensation while quantizing columns with searched MXFP4 block scales."""

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

    damp = damp_percent * torch.mean(torch.diag(hessian))
    hessian = hessian + torch.eye(columns, dtype=hessian.dtype, device=hessian.device) * damp
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


def gptq_quantize_weight_with_dynamic_mxfp4_scale_search(
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    exponent_offsets: list[int],
    scale_objective: str,
    damp_percent: float,
    device: torch.device | str,
    block_size: int = 32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run block-wise GPTQ while searching MXFP4 scales from each compensated block."""

    compute_device = torch.device(device)
    work_weight = weight.detach().to(device=compute_device, dtype=torch.float32).clone()
    hessian = gram.detach().to(device=compute_device, dtype=torch.float32).clone()
    rows, columns = work_weight.shape

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
    damp = damp_percent * torch.mean(torch.diag(hessian))
    hessian = hessian + torch.eye(columns, dtype=hessian.dtype, device=hessian.device) * damp
    hessian_inv = torch.linalg.cholesky(hessian)
    hessian_inv = torch.cholesky_inverse(hessian_inv)
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

    selected_offset_counts: dict[str, int] = {}
    block_diagonal_error = torch.zeros((), dtype=torch.float32, device=compute_device)
    default_block_diagonal_error = torch.zeros((), dtype=torch.float32, device=compute_device)

    for begin in range(0, columns, block_size):
        end = min(begin + block_size, columns)
        scale_search = search_mxfp4_hessian_scales(
            weight=work_weight[:, begin:end],
            gram=search_hessian[begin:end, begin:end],
            exponent_offsets=exponent_offsets,
            objective=scale_objective,
            device=compute_device,
            block_size=block_size,
        )
        quantized_block = scale_search.quantized_weight
        quantized_weight[:, begin:end] = quantized_block

        _merge_counts(selected_offset_counts, scale_search.metrics["selected_offset_counts"])
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
        "exponent_offsets": exponent_offsets,
        "scale_objective": scale_objective,
        "selected_offset_counts": selected_offset_counts,
        "dynamic_block_diagonal_error": block_diagonal_error.item(),
        "dynamic_default_block_diagonal_error": default_block_diagonal_error.item(),
        "dynamic_block_diagonal_error_ratio": (
            block_diagonal_error / default_block_diagonal_error.clamp_min(1e-12)
        ).item(),
        "gptq_granularity": "block",
    }


def _run_mxfp4_gptq_candidate(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    gram: torch.Tensor,
    activation_format: str,
    rotation: str,
    exponent_offsets: list[int],
    scale_objective: str,
    damp_percent: float,
    dynamic_scale_search: bool,
    device: torch.device | str,
) -> _MXFP4GPTQCandidate:
    compute_device = torch.device(device)
    rotated_weight = _rotate_weight(weight, rotation=rotation)
    rotated_gram = _rotate_gram(gram, rotation=rotation).to(device=compute_device)
    rotated_inputs = _rotate_weight(inputs, rotation=rotation)

    if dynamic_scale_search:
        optimized_weight, candidate_metrics = gptq_quantize_weight_with_dynamic_mxfp4_scale_search(
            weight=rotated_weight,
            gram=rotated_gram,
            exponent_offsets=exponent_offsets,
            scale_objective=scale_objective,
            damp_percent=damp_percent,
            device=compute_device,
        )
    else:
        scale_search = search_mxfp4_hessian_scales(
            weight=rotated_weight,
            gram=rotated_gram,
            exponent_offsets=exponent_offsets,
            objective=scale_objective,
            device=compute_device,
        )
        optimized_weight = gptq_quantize_weight_with_mxfp4_scales(
            weight=rotated_weight,
            gram=rotated_gram,
            selected_scales=scale_search.selected_scales,
            damp_percent=damp_percent,
            device=compute_device,
        )
        candidate_metrics = _prefix_metrics("scale_search", scale_search.metrics)

    weight_relative_error = _relative_reconstruction_error(
        original_weight=rotated_weight,
        quantized_weight=optimized_weight,
        gram=rotated_gram,
    )
    output_relative_error = _relative_output_error(
        original_weight=weight,
        quantized_weight=optimized_weight,
        bias=bias,
        inputs=inputs,
        rotated_inputs=rotated_inputs,
        activation_format=activation_format,
    )
    return _MXFP4GPTQCandidate(
        rotation=rotation,
        optimized_weight=optimized_weight,
        output_relative_error=output_relative_error,
        weight_relative_error=weight_relative_error,
        metrics={
            "rotation": rotation,
            "relative_reconstruction_error": weight_relative_error,
            **candidate_metrics,
        },
    )


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


def _relative_reconstruction_error(
    *,
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    gram: torch.Tensor,
) -> float:
    denominator = _quadratic_weight_error(original_weight, gram).clamp_min(1e-12)
    numerator = _quadratic_weight_error(original_weight - quantized_weight, gram)
    return (numerator / denominator).item()


def _relative_output_error(
    *,
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    rotated_inputs: torch.Tensor,
    activation_format: str,
) -> float:
    reference = F.linear(inputs, original_weight, bias)
    quantized_inputs = _quantize_activation(rotated_inputs, activation_format=activation_format)
    candidate = F.linear(quantized_inputs, quantized_weight, bias)
    denominator = torch.sum(reference.float().square()).clamp_min(1e-12)
    numerator = torch.sum((reference - candidate).float().square())
    return (numerator / denominator).item()


def _quantize_activation(inputs: torch.Tensor, *, activation_format: str) -> torch.Tensor:
    if activation_format == "none":
        return inputs
    return get_fp4_format(activation_format).quantize(inputs)


def _quadratic_weight_error(weight_error: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    gram = gram.detach().to(device=weight_error.device, dtype=torch.float32)
    weight_error = weight_error.float()
    return torch.sum((weight_error @ gram) * weight_error)
