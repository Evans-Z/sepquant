from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from tqdm import tqdm

from sepquant.calibration import load_layer_gram, load_layer_input
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class LayerOptimizationContext:
    layer_name: str
    module: nn.Linear
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
) -> list[LayerOptimizationResult]:
    """Run a layer optimizer while lazily loading each layer's calibration data."""

    results = []
    for target in tqdm(targets, desc=f"Optimizing layers ({optimizer.name})", unit="layer"):
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


def build_plan_from_results(
    *,
    results: list[LayerOptimizationResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "metadata": metadata,
        "layers": {
            result.layer_name: {
                "weight_format": result.spec.weight_format,
                "activation_format": result.spec.activation_format,
                "enabled": result.spec.enabled,
                **result.metrics,
            }
            for result in results
        },
    }

