from __future__ import annotations

from sepquant.optimization.methods.gptq import GPTQOptimizer
from sepquant.optimization.methods.hif4_scale_search_gptq import (
    HIF4DynamicScaleSearchGPTQOptimizer,
    HIF4ScaleSearchGPTQOptimizer,
)
from sepquant.optimization.methods.mxfp4_scale_search import MXFP4HessianScaleSearchOptimizer
from sepquant.optimization.methods.mxfp4_scale_search_gptq import (
    MXFP4DynamicScaleSearchGPTQOptimizer,
    MXFP4PlusDynamicScaleSearchGPTQOptimizer,
    MXFP4RotationSelectGPTQOptimizer,
    MXFP4ScaleSearchGPTQOptimizer,
)
from sepquant.optimization.methods.nvfp4_scale_search_gptq import (
    NVFP4DynamicScaleSearchGPTQOptimizer,
    NVFP4ScaleSearchGPTQOptimizer,
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
    mxfp4_plus_macro_scale_code_offsets: list[int] | None = None,
    mxfp4_scale_objective: str = "block",
    nvfp4_scale_code_offsets: list[int] | None = None,
    hif4_level1_code_offsets: list[int] | None = None,
    rotation: str = "none",
    device: str = "auto",
):
    if method == "weight_format_search":
        return WeightFormatSearchOptimizer(
            candidates=candidates,
            activation_format=activation_format,
            rotation=rotation,
        )
    if method == "gptq":
        return GPTQOptimizer(
            weight_format=weight_format,
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            rotation=rotation,
            device=device,
        )
    if method == "mxfp4_hessian_scale_search":
        if weight_format != "mxfp4":
            raise ValueError("mxfp4_hessian_scale_search only supports weight_format='mxfp4'")
        return MXFP4HessianScaleSearchOptimizer(
            activation_format=activation_format,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            objective=mxfp4_scale_objective,
            rotation=rotation,
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
            rotation=rotation,
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
            rotation=rotation,
            device=device,
        )
    if method == "mxfp4_plus_dynamic_scale_search_gptq":
        if weight_format != "mxfp4_plus":
            raise ValueError("mxfp4_plus_dynamic_scale_search_gptq only supports weight_format='mxfp4_plus'")
        return MXFP4PlusDynamicScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            macro_scale_code_offsets=mxfp4_plus_macro_scale_code_offsets
            or [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            scale_objective=mxfp4_scale_objective,
            rotation=rotation,
            device=device,
        )
    if method == "mxfp4_hessian_scale_search_gptq_rotation_select":
        if weight_format != "mxfp4":
            raise ValueError(
                "mxfp4_hessian_scale_search_gptq_rotation_select only supports weight_format='mxfp4'"
            )
        return MXFP4RotationSelectGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            candidate_rotation=rotation,
            dynamic_scale_search=False,
            name="mxfp4_hessian_scale_search_gptq_rotation_select",
            device=device,
        )
    if method == "mxfp4_dynamic_scale_search_gptq_rotation_select":
        if weight_format != "mxfp4":
            raise ValueError(
                "mxfp4_dynamic_scale_search_gptq_rotation_select only supports weight_format='mxfp4'"
            )
        return MXFP4RotationSelectGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            exponent_offsets=mxfp4_scale_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            candidate_rotation=rotation,
            dynamic_scale_search=True,
            name="mxfp4_dynamic_scale_search_gptq_rotation_select",
            device=device,
        )
    if method == "nvfp4_hessian_scale_search_gptq":
        if weight_format != "nvfp4":
            raise ValueError("nvfp4_hessian_scale_search_gptq only supports weight_format='nvfp4'")
        return NVFP4ScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            scale_code_offsets=nvfp4_scale_code_offsets or [-3, -2, -1, 0, 1, 2, 3],
            scale_objective=mxfp4_scale_objective,
            rotation=rotation,
            device=device,
        )
    if method == "nvfp4_dynamic_scale_search_gptq":
        if weight_format != "nvfp4":
            raise ValueError("nvfp4_dynamic_scale_search_gptq only supports weight_format='nvfp4'")
        return NVFP4DynamicScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            scale_code_offsets=nvfp4_scale_code_offsets or [-3, -2, -1, 0, 1, 2, 3],
            scale_objective=mxfp4_scale_objective,
            rotation=rotation,
            device=device,
        )
    if method == "hif4_hessian_scale_search_gptq":
        if weight_format != "hif4":
            raise ValueError("hif4_hessian_scale_search_gptq only supports weight_format='hif4'")
        return HIF4ScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            level1_code_offsets=hif4_level1_code_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            rotation=rotation,
            device=device,
        )
    if method == "hif4_dynamic_scale_search_gptq":
        if weight_format != "hif4":
            raise ValueError("hif4_dynamic_scale_search_gptq only supports weight_format='hif4'")
        return HIF4DynamicScaleSearchGPTQOptimizer(
            activation_format=activation_format,
            damp_percent=gptq_damp_percent,
            level1_code_offsets=hif4_level1_code_offsets or [-2, -1, 0, 1, 2],
            scale_objective=mxfp4_scale_objective,
            rotation=rotation,
            device=device,
        )
    raise ValueError(f"Unsupported optimization method: {method}")


__all__ = [
    "GPTQOptimizer",
    "HIF4DynamicScaleSearchGPTQOptimizer",
    "HIF4ScaleSearchGPTQOptimizer",
    "MXFP4DynamicScaleSearchGPTQOptimizer",
    "MXFP4HessianScaleSearchOptimizer",
    "MXFP4PlusDynamicScaleSearchGPTQOptimizer",
    "MXFP4RotationSelectGPTQOptimizer",
    "MXFP4ScaleSearchGPTQOptimizer",
    "NVFP4DynamicScaleSearchGPTQOptimizer",
    "NVFP4ScaleSearchGPTQOptimizer",
    "WeightFormatSearchOptimizer",
    "build_layer_optimizer",
]
