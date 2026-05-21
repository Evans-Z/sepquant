from __future__ import annotations

from dataclasses import dataclass

import torch

from sepquant.formats import get_fp4_format
from sepquant.optimization.layerwise import LayerOptimizationContext, LayerOptimizationResult
from sepquant.quantization import LayerQuantizationSpec


@dataclass(frozen=True)
class FormatSearchResult:
    layer_name: str
    best_format: str
    errors: dict[str, float]


@dataclass(frozen=True)
class WeightFormatSearchOptimizer:
    candidates: list[str]
    activation_format: str = "none"
    rotation: str = "none"
    name: str = "weight_format_search"

    def optimize(self, context: LayerOptimizationContext) -> LayerOptimizationResult:
        if context.gram is not None:
            result = search_layer_weight_format_from_gram(
                layer_name=context.layer_name,
                module=context.module,
                gram=context.gram,
                candidates=self.candidates,
            )
        elif context.inputs is not None:
            result = search_layer_weight_format(
                layer_name=context.layer_name,
                module=context.module,
                inputs=context.inputs,
                candidates=self.candidates,
            )
        else:
            return LayerOptimizationResult(
                layer_name=context.layer_name,
                spec=LayerQuantizationSpec(enabled=False),
                metrics={"reason": "missing_calibration"},
            )

        return LayerOptimizationResult(
            layer_name=context.layer_name,
            spec=LayerQuantizationSpec(
                weight_format=result.best_format,
                activation_format=self.activation_format,
                rotation=self.rotation,
                enabled=True,
            ),
            metrics={"errors": result.errors},
        )


def search_layer_weight_format(
    *,
    layer_name: str,
    module: torch.nn.Linear,
    inputs: torch.Tensor,
    candidates: list[str],
) -> FormatSearchResult:
    weight = module.weight.detach().float().cpu()
    reference = inputs @ weight.t()
    denominator = torch.mean(reference.square()).clamp_min(1e-12)

    errors: dict[str, float] = {}
    for candidate in candidates:
        quantized_weight = get_fp4_format(candidate).quantize(weight)
        candidate_output = inputs @ quantized_weight.t()
        rel_mse = torch.mean((reference - candidate_output).square()) / denominator
        errors[candidate] = rel_mse.item()

    best_format = min(errors, key=errors.get)
    return FormatSearchResult(layer_name=layer_name, best_format=best_format, errors=errors)


def search_layer_weight_format_from_gram(
    *,
    layer_name: str,
    module: torch.nn.Linear,
    gram: torch.Tensor,
    candidates: list[str],
) -> FormatSearchResult:
    weight = module.weight.detach().float().cpu()
    gram = gram.float().cpu()
    denominator = _quadratic_weight_error(weight, gram).clamp_min(1e-12)

    errors: dict[str, float] = {}
    for candidate in candidates:
        quantized_weight = get_fp4_format(candidate).quantize(weight)
        error_weight = weight - quantized_weight
        rel_mse = _quadratic_weight_error(error_weight, gram) / denominator
        errors[candidate] = rel_mse.item()

    best_format = min(errors, key=errors.get)
    return FormatSearchResult(layer_name=layer_name, best_format=best_format, errors=errors)


def _quadratic_weight_error(weight_error: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    return torch.sum((weight_error @ gram) * weight_error)
