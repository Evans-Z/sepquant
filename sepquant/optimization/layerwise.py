from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from tqdm import tqdm

from sepquant.calibration import load_calibration_metadata, load_layer_gram, load_layer_input
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class LayerOptimizationContext:
    layer_name: str
    module: nn.Module
    gram: torch.Tensor | None = None
    inputs: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerOptimizationResult:
    layer_name: str
    spec: LayerQuantizationSpec
    metrics: dict[str, Any] = field(default_factory=dict)
    optimized_weight: torch.Tensor | None = None


class LayerOptimizer(Protocol):
    name: str

    def optimize(self, context: LayerOptimizationContext) -> LayerOptimizationResult:
        """Optimize a single layer and return a quantization decision."""
        ...


def optimize_layers(
    *,
    targets: list,
    calibration_dir: str | Path,
    optimizer: LayerOptimizer,
    require_calibration: bool = True,
    min_tokens_per_layer: int = 0,
    fallback_weight_format: str | None = None,
    fallback_activation_format: str | None = None,
    fallback_rotation: str | None = None,
) -> list[LayerOptimizationResult]:
    """Run a layer optimizer while lazily loading each layer's calibration data."""

    if min_tokens_per_layer < 0:
        raise ValueError("min_tokens_per_layer must be non-negative")

    metadata = load_calibration_metadata(calibration_dir) if min_tokens_per_layer > 0 else {}
    token_counts = metadata.get("token_counts", {})
    if not isinstance(token_counts, dict):
        token_counts = {}

    results = []
    for target in tqdm(targets, desc=f"Optimizing layers ({optimizer.name})", unit="layer"):
        token_count = int(token_counts.get(target.name, 0))
        if min_tokens_per_layer > 0 and token_count < min_tokens_per_layer:
            fallback = _fallback_result(
                layer_name=target.name,
                token_count=token_count,
                min_tokens_per_layer=min_tokens_per_layer,
                fallback_weight_format=fallback_weight_format,
                fallback_activation_format=fallback_activation_format,
                fallback_rotation=fallback_rotation,
            )
            if fallback is not None:
                results.append(fallback)
            continue

        gram = load_layer_gram(calibration_dir, target.name)
        inputs = load_layer_input(calibration_dir, target.name)
        if require_calibration and gram is None and inputs is None:
            continue

        result = optimizer.optimize(
            LayerOptimizationContext(
                layer_name=target.name,
                module=target.module,
                gram=gram,
                inputs=inputs,
            )
        )
        results.append(result)
    return results


def _fallback_result(
    *,
    layer_name: str,
    token_count: int,
    min_tokens_per_layer: int,
    fallback_weight_format: str | None,
    fallback_activation_format: str | None,
    fallback_rotation: str | None,
) -> LayerOptimizationResult | None:
    if fallback_weight_format is None:
        return None

    enabled = fallback_weight_format != "none"
    return LayerOptimizationResult(
        layer_name=layer_name,
        spec=LayerQuantizationSpec(
            weight_format=None if fallback_weight_format == "none" else fallback_weight_format,
            activation_format=fallback_activation_format,
            rotation=fallback_rotation,
            enabled=enabled,
        ),
        metrics={
            "method": "fallback",
            "reason": "insufficient_calibration_tokens",
            "token_count": token_count,
            "min_tokens_per_layer": min_tokens_per_layer,
            "fallback_weight_format": fallback_weight_format,
        },
    )


def build_plan_from_results(
    *,
    results: list[LayerOptimizationResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    layers = {
        result.layer_name: {
            "weight_format": result.spec.weight_format,
            "activation_format": result.spec.activation_format,
            "rotation": result.spec.rotation,
            "enabled": result.spec.enabled,
            **result.metrics,
        }
        for result in results
    }
    return {
        "version": 1,
        "metadata": metadata,
        "summary": _build_plan_summary(results),
        "layers": layers,
    }


def _build_plan_summary(results: list[LayerOptimizationResult]) -> dict[str, Any]:
    enabled_results = [result for result in results if result.spec.enabled]
    return {
        "total_layers": len(results),
        "enabled_layers": len(enabled_results),
        "disabled_layers": len(results) - len(enabled_results),
        "rotation_counts": _count_spec_field(enabled_results, "rotation"),
        "weight_format_counts": _count_spec_field(enabled_results, "weight_format"),
        "activation_format_counts": _count_spec_field(enabled_results, "activation_format"),
    }


def _count_spec_field(results: list[LayerOptimizationResult], field_name: str) -> dict[str, int]:
    counts = Counter(_format_summary_value(getattr(result.spec, field_name)) for result in results)
    return dict(sorted(counts.items()))


def _format_summary_value(value: Any) -> str:
    return "none" if value is None else str(value)

