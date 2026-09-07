import torch
from torch import nn

from sepquant.optimization.layerwise import LayerOptimizationContext
from sepquant.optimization.methods import build_layer_optimizer
from sepquant.optimization.methods.mxfp4_float_scale_search_gptq import (
    MXFP4_E4M3_SCALE_SPEC,
    MXFP4_E5M3_SCALE_SPEC,
    search_mxfp4_float_hessian_scales,
)


def test_search_mxfp4_e4m3_hessian_scales_uses_code_offsets() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 65)
    inputs = torch.randn(64, 65)
    gram = inputs.t().matmul(inputs)

    result = search_mxfp4_float_hessian_scales(
        weight=weight,
        gram=gram,
        scale_spec=MXFP4_E4M3_SCALE_SPEC,
        scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
        objective="diag",
        device="cpu",
    )

    assert result.quantized_weight.shape == weight.shape
    assert result.selected_scales.shape == (8, 3)
    assert result.selected_scale_code_offsets.shape == (8, 3)
    assert result.metrics["block_size"] == 32
    assert result.metrics["weight_format"] == "mxfp4_e4m3"
    assert result.metrics["scale_code_offsets"] == [-3, -2, -1, 0, 1, 2, 3]
    assert result.metrics["block_diagonal_error_ratio"] <= 1.0


def test_search_mxfp4_e5m3_hessian_scales_uses_code_offsets() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 65)
    inputs = torch.randn(64, 65)
    gram = inputs.t().matmul(inputs)

    result = search_mxfp4_float_hessian_scales(
        weight=weight,
        gram=gram,
        scale_spec=MXFP4_E5M3_SCALE_SPEC,
        scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
        objective="diag",
        device="cpu",
    )

    assert result.quantized_weight.shape == weight.shape
    assert result.metrics["weight_format"] == "mxfp4_e5m3"
    assert result.metrics["scale_exponent_bits"] == 5
    assert result.metrics["block_diagonal_error_ratio"] <= 1.0


def test_mxfp4_e4m3_dynamic_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 16)
    inputs = torch.randn(64, 64)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="mxfp4_e4m3_dynamic_scale_search_gptq",
        candidates=[],
        weight_format="mxfp4_e4m3",
        activation_format="mxfp4_e4m3_search",
        mxfp4_float_scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
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
    assert result.spec.weight_format == "mxfp4_e4m3"
    assert result.spec.activation_format == "mxfp4_e4m3_search"
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert result.metrics["block_size"] == 32
    assert result.metrics["weight_format"] == "mxfp4_e4m3"


def test_mxfp4_e5m3_dynamic_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 16)
    inputs = torch.randn(64, 64)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="mxfp4_e5m3_dynamic_scale_search_gptq",
        candidates=[],
        weight_format="mxfp4_e5m3",
        activation_format="mxfp4_e5m3_search",
        mxfp4_float_scale_code_offsets=[-3, -2, -1, 0, 1, 2, 3],
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
    assert result.spec.weight_format == "mxfp4_e5m3"
    assert result.spec.activation_format == "mxfp4_e5m3_search"
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert result.metrics["scale_exponent_bits"] == 5


def test_mxfp4_e4m3_hessian_scale_search_gptq_optimizer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 8)
    inputs = torch.randn(32, 64)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="mxfp4_e4m3_hessian_scale_search_gptq",
        candidates=[],
        weight_format="mxfp4_e4m3",
        activation_format="none",
        mxfp4_float_scale_code_offsets=[-2, -1, 0, 1, 2],
        mxfp4_scale_objective="diag",
        device="cpu",
    )

    result = optimizer.optimize(
        LayerOptimizationContext(
            layer_name="layers.0.mlp.up_proj",
            module=linear,
            gram=gram,
            inputs=None,
        )
    )

    assert result.spec.enabled
    assert result.spec.weight_format == "mxfp4_e4m3"
    assert result.optimized_weight is not None
    assert "scale_search_block_diagonal_error_ratio" in result.metrics
