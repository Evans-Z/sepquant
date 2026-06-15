from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

from sepquant.formats import FP4Format, get_fp4_format
from sepquant.models.quant_linear import QuantLinear
from sepquant.quantization import LayerQuantizationSpec, QuantizationPlan


ModelType = Literal[
    "auto",
    "qwen",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "llama",
    "mistral",
    "gemma",
    "opt",
    "generic",
]


@dataclass(frozen=True)
class PatchReport:
    replaced: int
    skipped: tuple[str, ...]
    model_type: str


@dataclass(frozen=True)
class TargetLinear:
    name: str
    module: nn.Linear


TRANSFORMER_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

OPT_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
)

SUPPORTED_SUFFIX_MODEL_TYPES = {
    "qwen",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "llama",
    "mistral",
    "gemma",
    "opt",
}


def patch_causal_lm_linears(
    model: nn.Module,
    *,
    weight_format: FP4Format | None,
    activation_format: FP4Format | None = None,
    model_type: ModelType = "auto",
    include_lm_head: bool = False,
    quantization_plan: QuantizationPlan | None = None,
    prequantized_weight: bool = False,
    rotation: str = "none",
) -> PatchReport:
    """Replace supported causal LM linear projections with `QuantLinear`."""

    resolved_model_type, targets = get_target_linears(
        model,
        model_type=model_type,
        include_lm_head=include_lm_head,
    )
    skipped: list[str] = []

    for target in targets:
        name = target.name
        module = target.module
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        if not isinstance(module, nn.Linear):
            skipped.append(name)
            continue

        layer_weight_format, layer_activation_format, layer_rotation = _resolve_layer_formats(
            layer_name=name,
            fallback_weight_format=weight_format,
            fallback_activation_format=activation_format,
            fallback_rotation=rotation,
            quantization_plan=quantization_plan,
        )
        if layer_weight_format is None:
            skipped.append(name)
            continue

        quant_linear = (
            QuantLinear.from_prequantized(
                module,
                weight_format=layer_weight_format,
                activation_format=layer_activation_format,
                rotation=layer_rotation,
            )
            if prequantized_weight
            else QuantLinear.from_float(
                module,
                weight_format=layer_weight_format,
                activation_format=layer_activation_format,
                rotation=layer_rotation,
            )
        )
        setattr(parent, child_name, quant_linear)

    return PatchReport(
        replaced=len(targets) - len(skipped),
        skipped=tuple(skipped),
        model_type=resolved_model_type,
    )


def _resolve_model_type(model: nn.Module, model_type: ModelType) -> str:
    if model_type != "auto":
        return model_type

    config_model_type = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(config_model_type, str):
        normalized = config_model_type.lower().replace("-", "_")
        if normalized in SUPPORTED_SUFFIX_MODEL_TYPES:
            return normalized
    return "generic"


def get_target_linears(
    model: nn.Module,
    *,
    model_type: ModelType = "auto",
    include_lm_head: bool,
) -> tuple[str, list[TargetLinear]]:
    resolved_model_type = _resolve_model_type(model, model_type)
    targets: list[TargetLinear] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if name == "lm_head" and not include_lm_head:
            continue

        if resolved_model_type == "generic":
            targets.append(TargetLinear(name=name, module=module))
        elif resolved_model_type == "opt":
            if name.endswith(OPT_LINEAR_SUFFIXES) or (include_lm_head and name == "lm_head"):
                targets.append(TargetLinear(name=name, module=module))
        elif name.endswith(TRANSFORMER_LINEAR_SUFFIXES) or (include_lm_head and name == "lm_head"):
            targets.append(TargetLinear(name=name, module=module))

    return resolved_model_type, targets


def _resolve_layer_formats(
    *,
    layer_name: str,
    fallback_weight_format: FP4Format | None,
    fallback_activation_format: FP4Format | None,
    fallback_rotation: str,
    quantization_plan: QuantizationPlan | None,
) -> tuple[FP4Format | None, FP4Format | None, str]:
    if quantization_plan is None:
        return fallback_weight_format, fallback_activation_format, fallback_rotation

    spec = quantization_plan.get(layer_name)
    if spec is None:
        return fallback_weight_format, fallback_activation_format, fallback_rotation
    if not spec.enabled:
        return None, None, fallback_rotation

    weight_format = _format_from_spec(
        spec=spec,
        field_name="weight_format",
        fallback=fallback_weight_format,
    )
    activation_format = _format_from_spec(
        spec=spec,
        field_name="activation_format",
        fallback=fallback_activation_format,
    )
    rotation = spec.rotation if spec.rotation is not None else fallback_rotation
    return weight_format, activation_format, rotation


def _format_from_spec(
    *,
    spec: LayerQuantizationSpec,
    field_name: str,
    fallback: FP4Format | None,
) -> FP4Format | None:
    format_name = getattr(spec, field_name)
    if format_name is None:
        return fallback
    if format_name == "none":
        return None
    return get_fp4_format(format_name)

