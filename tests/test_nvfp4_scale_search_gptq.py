import torch
from torch import nn

from sepquant.optimization.layerwise import LayerOptimizationContext
from sepquant.optimization.methods import build_layer_optimizer
from sepquant.optimization.methods.nvfp4_scale_search_gptq import (
    search_nvfp4_hessian_scales,
)


def test_search_nvfp4_hessian_scales_uses_e4m3_code_offsets() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 33)
    inputs = torch.randn(64, 33)
    gram = inputs.t().matmul(inputs)

    result = search_nvfp4_hessian_scales(
        weight=weight,
        gram=gram,
        scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
        objective="diag",
        device="cpu",
    )

    assert result.quantized_weight.shape == weight.shape
    assert result.selected_scales.shape == (8, 3)
    assert result.selected_scale_code_offsets.shape == (8, 3)
    assert result.metrics["block_size"] == 16
    assert result.metrics["scale_code_offsets"] == [-3, -2, -1, 0, 1, 2, 3]
    assert result.metrics["block_diagonal_error_ratio"] <= 1.0


def test_nvfp4_dynamic_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(32, 16)
    inputs = torch.randn(64, 32)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="nvfp4_dynamic_scale_search_gptq",
        candidates=[],
        weight_format="nvfp4",
        activation_format="nvfp4_search",
        nvfp4_scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
        device="cpu",
    )

    result = optimizer.optimize(
        LayerOptimizationContext(
            layer_name="layers.0.mlp.down_proj",
            module=linear,
            gram=gram,
            inputs=None,
        )
    )

    assert result.spec.enabled
    assert result.spec.weight_format == "nvfp4"
    assert result.spec.activation_format == "nvfp4_search"
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert result.metrics["block_size"] == 16
