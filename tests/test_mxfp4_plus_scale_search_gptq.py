import torch
from torch import nn

from sepquant.optimization.layerwise import LayerOptimizationContext
from sepquant.optimization.methods import build_layer_optimizer
from sepquant.optimization.methods.mxfp4_scale_search_gptq import (
    search_mxfp4_plus_hessian_scales,
)


def test_search_mxfp4_plus_hessian_scales_uses_macro_scale_code_offsets() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 129)
    inputs = torch.randn(64, 129)
    gram = inputs.t().matmul(inputs)

    result = search_mxfp4_plus_hessian_scales(
        weight=weight,
        gram=gram,
        macro_scale_code_offsets=[-4, -2, 0, 2, 4],
        objective="diag",
        device="cpu",
    )

    assert result.quantized_weight.shape == weight.shape
    assert result.selected_macro_scale_codes.shape == (8, 2)
    assert result.selected_macro_scale_code_offsets.shape == (8, 2)
    assert result.metrics["block_size"] == 16
    assert result.metrics["macro_block_size"] == 128
    assert result.metrics["macro_scale_code_offsets"] == [-4, -2, 0, 2, 4]
    assert result.metrics["block_diagonal_error_ratio"] <= 1.0


def test_mxfp4_plus_dynamic_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(128, 16)
    inputs = torch.randn(64, 128)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="mxfp4_plus_dynamic_scale_search_gptq",
        candidates=[],
        weight_format="mxfp4_plus",
        activation_format="mxfp4_plus_search",
        mxfp4_plus_macro_scale_code_offsets=[-4, -2, 0, 2, 4],
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
    assert result.spec.weight_format == "mxfp4_plus"
    assert result.spec.activation_format == "mxfp4_plus_search"
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert result.metrics["block_size"] == 16
    assert result.metrics["macro_block_size"] == 128
