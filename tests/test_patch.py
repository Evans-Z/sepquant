import torch
from torch import nn

from sepquant.formats import get_fp4_format
from sepquant.models import QuantLinear, patch_causal_lm_linears
from sepquant.quantization import QuantizationPlan


class TinyQwenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)


class TinyOptBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.down_proj = nn.Linear(8, 8)


def test_patch_qwen_linears_uses_generic_suffix_filter() -> None:
    model = TinyQwenBlock()

    report = patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        model_type="qwen3",
    )

    assert report.replaced == 2
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.k_proj, QuantLinear)
    assert isinstance(model.other, nn.Linear)


def test_patch_opt_linears_uses_opt_suffix_filter() -> None:
    model = TinyOptBlock()

    report = patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        model_type="opt",
    )

    assert report.replaced == 4
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.out_proj, QuantLinear)
    assert isinstance(model.fc1, QuantLinear)
    assert isinstance(model.fc2, QuantLinear)
    assert isinstance(model.down_proj, nn.Linear)


def test_patch_linears_uses_quantization_plan() -> None:
    model = TinyOptBlock()
    plan = QuantizationPlan.from_dict(
        {
            "default": {"weight_format": "mxfp4", "activation_format": "none"},
            "layers": {"fc2": {"enabled": False}},
            "patterns": [{"pattern": "*fc1", "weight_format": "nvfp4"}],
        }
    )

    report = patch_causal_lm_linears(
        model,
        weight_format=None,
        model_type="opt",
        quantization_plan=plan,
    )

    assert report.replaced == 3
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.out_proj, QuantLinear)
    assert isinstance(model.fc1, QuantLinear)
    assert model.fc1.weight_format.name == "nvfp4"
    assert isinstance(model.fc2, nn.Linear)


def test_quant_linear_forward_shape() -> None:
    linear = nn.Linear(8, 4)
    quant_linear = QuantLinear.from_float(
        linear,
        weight_format=get_fp4_format("mxfp4"),
        activation_format=get_fp4_format("nvfp4"),
    )

    output = quant_linear(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 4)


def test_prequantized_quant_linear_preserves_weight() -> None:
    linear = nn.Linear(8, 4)
    original_weight = linear.weight.detach().clone()

    quant_linear = QuantLinear.from_prequantized(
        linear,
        weight_format=get_fp4_format("mxfp4"),
        activation_format=get_fp4_format("mxfp4_search"),
    )

    assert torch.equal(quant_linear.weight, original_weight)
    assert quant_linear.activation_format is not None
    assert quant_linear.activation_format.name == "mxfp4_search"

