from __future__ import annotations

from sepquant.optimization.methods.gptq import GPTQOptimizer
from sepquant.optimization.methods.mxfp4_scale_search import MXFP4HessianScaleSearchOptimizer
from sepquant.optimization.methods.mxfp4_scale_search_gptq import (
    MXFP4DynamicScaleSearchGPTQOptimizer,
    MXFP4ScaleSearchGPTQOptimizer,
)
from sepquant.optimization.methods.weight_format_search import WeightFormatSearchOptimizer


def build_layer_optimizer(
    *,
    method: str,
    candidates: list[str],
    activation_format: str,
    weight_format: str = "mxfp4",
    gptq_damp_percent: float = 0.01,
    mxfp4_scale_offsets: list[int] | None = None,
    mxfp4_scale_objective: str = "block",
    device: str = "auto",
):
    if method == "weight_format_search":
        return WeightFormatSearchOptimizer(
            candidates=candidates,
            activation_format=activation_format,
        )
    if method == "gptq":
        return GPTQOptimizer(
            weight_format=weight_format,
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            device=device,
        )
    if method == "mxfp4_hessian_scale_search":
        if weight_format != "mxfp4":
            raise ValueError("mxfp4_hessian_scale_search only supports weight_format='mxfp4'")
        return MXFP4HessianScaleSearchOptimizer(
            activation_format=activation_format,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            objective=mxfp4_scale_objective,
            device=device,
        )
    if method == "mxfp4_hessian_scale_search_gptq":
        if weight_format != "mxfp4":
            raise ValueError("mxfp4_hessian_scale_search_gptq only supports weight_format='mxfp4'")
        return MXFP4ScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            device=device,
        )
    if method == "mxfp4_dynamic_scale_search_gptq":
        if weight_format != "mxfp4":
            raise ValueError("mxfp4_dynamic_scale_search_gptq only supports weight_format='mxfp4'")
        return MXFP4DynamicScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            device=device,
        )
    raise ValueError(f"Unsupported optimization method: {method}")


__all__ = [
    "GPTQOptimizer",
    "MXFP4DynamicScaleSearchGPTQOptimizer",
    "MXFP4HessianScaleSearchOptimizer",
    "MXFP4ScaleSearchGPTQOptimizer",
    "WeightFormatSearchOptimizer",
    "build_layer_optimizer",
]
