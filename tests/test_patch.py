import torch
from torch import nn

from sepquant.calibration.capture import collect_linear_calibration
from sepquant.formats import get_fp4_format
from sepquant.models import QuantLinear, get_target_linears, patch_causal_lm_linears
from sepquant.quantization import QuantizationPlan


class CountingFormat:
    name = "counting"
    block_size = 1

    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(tensor.shape))
        return tensor


class TinyQwenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)


class TinyQwen3MoeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(8, 2, bias=False)
        self.experts = TinyQwen3MoeExperts()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        top_k_index = torch.arange(token_count, device=hidden_states.device).remainder(2).unsqueeze(-1)
        top_k_weights = torch.ones(token_count, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        return self.experts(hidden_states, top_k_index, top_k_weights)


class TinyQwen3MoeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.randn(2, 32, 8))
        self.down_proj = nn.Parameter(torch.randn(2, 8, 16))
        self.act_fn = torch.relu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class TinyQwen3MoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "qwen3_moe"})()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.mlp = TinyQwen3MoeMlp()

    @property
    def device(self) -> torch.device:
        return self.q_proj.weight.device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class TinyQwen3MoeDenseMlpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "qwen3_moe"})()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.mlp = TinyQwen3MoeDenseMlp()


class TinyQwen3MoeDenseMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class TinyOptBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.down_proj = nn.Linear(8, 8)


class TinyRotationOptBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(32, 16)
        self.out_proj = nn.Linear(32, 16)


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


def test_patch_qwen3_moe_auto_uses_suffix_filter_and_skips_routers() -> None:
    model = TinyQwen3MoeBlock()
    resolved_model_type, targets = get_target_linears(model, model_type="auto", include_lm_head=False)

    report = patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        model_type="auto",
    )

    assert report.model_type == "qwen3_moe"
    assert resolved_model_type == "qwen3_moe"
    assert {target.name for target in targets} == {
        "q_proj",
        "o_proj",
        "mlp.experts.0.gate_up_proj",
        "mlp.experts.0.down_proj",
        "mlp.experts.1.gate_up_proj",
        "mlp.experts.1.down_proj",
    }
    assert report.replaced == 6
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.o_proj, QuantLinear)
    assert isinstance(model.mlp.gate, nn.Linear)


def test_collect_qwen3_moe_expert_calibration_counts_routed_tokens() -> None:
    model = TinyQwen3MoeBlock()
    _, targets = get_target_linears(model, model_type="auto", include_lm_head=False)

    capture = collect_linear_calibration(
        model=model,
        targets=targets,
        batches=[torch.randn(4, 8)],
        max_tokens_per_layer=16,
        capture_mode="gram",
    )

    assert capture.token_counts["mlp.experts.0.gate_up_proj"] == 2
    assert capture.token_counts["mlp.experts.0.down_proj"] == 2
    assert capture.token_counts["mlp.experts.1.gate_up_proj"] == 2
    assert capture.token_counts["mlp.experts.1.down_proj"] == 2
    assert capture.grams["mlp.experts.0.gate_up_proj"].shape == (8, 8)
    assert capture.grams["mlp.experts.0.down_proj"].shape == (16, 16)


def test_patch_qwen3_moe_sparse_experts_quantizes_activations() -> None:
    model = TinyQwen3MoeBlock()
    activation_format = CountingFormat()

    patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        activation_format=activation_format,
        model_type="auto",
    )
    model(torch.randn(4, 8))

    assert sorted(activation_format.input_shapes) == [
        (2, 8),
        (2, 8),
        (2, 16),
        (2, 16),
    ]


def test_patch_qwen3_moe_dense_mlp_linears() -> None:
    model = TinyQwen3MoeDenseMlpBlock()
    resolved_model_type, targets = get_target_linears(model, model_type="auto", include_lm_head=False)

    report = patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        model_type="auto",
    )

    assert resolved_model_type == "qwen3_moe"
    assert {target.name for target in targets} == {
        "q_proj",
        "o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    assert report.replaced == 5
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.o_proj, QuantLinear)
    assert isinstance(model.mlp.gate_proj, QuantLinear)
    assert isinstance(model.mlp.up_proj, QuantLinear)
    assert isinstance(model.mlp.down_proj, QuantLinear)


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


def test_block_hadamard_rotation_preserves_forward_shape() -> None:
    linear = nn.Linear(8, 4)
    quant_linear = QuantLinear.from_float(
        linear,
        weight_format=get_fp4_format("mxfp4"),
        activation_format=get_fp4_format("mxfp4_search"),
        rotation="block_hadamard",
    )

    output = quant_linear(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 4)
    assert quant_linear.rotation == "block_hadamard"


def test_quantization_plan_controls_layer_rotation() -> None:
    model = TinyRotationOptBlock()
    plan = QuantizationPlan.from_dict(
        {
            "layers": {
                "q_proj": {
                    "weight_format": "mxfp4",
                    "activation_format": "mxfp4_search",
                    "rotation": "block_hadamard",
                },
                "out_proj": {
                    "weight_format": "mxfp4",
                    "activation_format": "mxfp4_search",
                    "rotation": "none",
                },
            }
        }
    )

    report = patch_causal_lm_linears(
        model,
        weight_format=None,
        activation_format=None,
        model_type="opt",
        quantization_plan=plan,
    )

    assert report.replaced == 2
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.out_proj, QuantLinear)
    assert model.q_proj.rotation == "block_hadamard"
    assert model.out_proj.rotation == "none"

