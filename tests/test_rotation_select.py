import torch
from torch import nn

from sepquant.optimization.layerwise import LayerOptimizationContext
from sepquant.optimization.methods import build_layer_optimizer


def test_rotation_select_optimizer_scores_full_candidates() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(32, 16)
    inputs = torch.randn(64, 32)
    gram = inputs.t().matmul(inputs)
    optimizer = build_layer_optimizer(
        method="mxfp4_dynamic_scale_search_gptq_rotation_select",
        candidates=[],
        activation_format="mxfp4_search",
        rotation="block_hadamard",
        device="cpu",
    )

    result = optimizer.optimize(
        LayerOptimizationContext(
            layer_name="layers.0.mlp.down_proj",
            module=linear,
            gram=gram,
            inputs=inputs,
        )
    )

    assert result.spec.enabled
    assert result.spec.weight_format == "mxfp4"
    assert result.spec.activation_format == "mxfp4_search"
    assert result.spec.rotation in {"none", "block_hadamard"}
    assert result.optimized_weight is not None
    assert result.optimized_weight.shape == linear.weight.shape
    assert "no_rotation_output_relative_error" in result.metrics
    assert "rotated_output_relative_error" in result.metrics


def test_rotation_select_requires_inputs() -> None:
    linear = nn.Linear(32, 16)
    optimizer = build_layer_optimizer(
        method="mxfp4_dynamic_scale_search_gptq_rotation_select",
        candidates=[],
        activation_format="mxfp4_search",
        rotation="block_hadamard",
        device="cpu",
    )

    result = optimizer.optimize(
        LayerOptimizationContext(
            layer_name="layers.0.mlp.down_proj",
            module=linear,
            gram=torch.eye(32),
            inputs=None,
        )
    )

    assert not result.spec.enabled
    assert result.metrics["reason"] == "missing_inputs"
