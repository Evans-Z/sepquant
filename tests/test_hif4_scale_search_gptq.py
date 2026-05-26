import torch
from torch import nn

from sepquant.optimization.layerwise import LayerOptimizationContext
from sepquant.optimization.methods import build_layer_optimizer
from sepquant.optimization.methods.hif4_scale_search_gptq import (
    search_hif4_hessian_scales,
)


def test_search_hif4_hessian_scales_uses_level1_code_offsets() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 129)
    inputs = torch.randn(64, 129)
    gram = inputs.t().matmul(inputs)

    result = search_hif4_hessian_scales(
        weight=weight,
        gram=gram,
        level1_code_offsets=[-2, -1, 0, 1, 2],
        objective="diag",
        device="cpu",
    )

    assert result.quantized_weight.shape == weight.shape
    assert result.selected_level1_codes.shape == (8, 3)
    assert result.selected_level1_code_offsets.shape == (8, 3)
    assert result.metrics["block_size"] == 64
    assert result.metrics["level1_code_offsets"] == [-2, -1, 0, 1, 2]
    assert result.metrics["block_diagonal_error_ratio"] <= 1.0


def test_hif4_dynamic_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(128, 16)
    inputs = torch.randn(64, 128)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="hif4_dynamic_scale_search_gptq",
        candidates=[],
        weight_format="hif4",
        activation_format="hif4_search",
        hif4_level1_code_offsets=[-2, -1, 0, 1, 2],
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
    assert result.spec.weight_format == "hif4"
    assert result.spec.activation_format == "hif4_search"
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert result.metrics["block_size"] == 64
