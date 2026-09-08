import pytest
import torch
from torch import nn

from sepquant.calibration import build_qwen3_vl_calibration_batch, collect_linear_calibration
from sepquant.formats import get_fp4_format
from sepquant.models import (
    QuantLinear,
    get_target_linears,
    patch_causal_lm_linears,
    patch_loaded_qwen3_vl,
)


class Namespace(nn.Module):
    pass


class TinyQwen3VL(nn.Module):
    def __init__(self, model_type: str = "qwen3_vl") -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": model_type})()
        self.model = Namespace()
        self.model.visual = Namespace()
        self.model.visual.blocks = nn.ModuleList([TinyVisionBlock()])
        self.model.visual.merger = TinyMerger()
        self.model.visual.deepstack_merger_list = nn.ModuleList([TinyMerger()])
        self.model.language_model = Namespace()
        self.model.language_model.layers = nn.ModuleList([TinyLanguageBlock()])
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual.blocks[0](pixel_values)
        visual = self.model.visual.merger(visual)
        visual = self.model.visual.deepstack_merger_list[0](visual)
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
        return self.model.language_model.layers[0](hidden) + visual.mean()


class TinyVisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Namespace()
        self.attn.qkv = nn.Linear(8, 8)
        self.attn.proj = nn.Linear(8, 8)
        self.mlp = Namespace()
        self.mlp.linear_fc1 = nn.Linear(8, 16)
        self.mlp.linear_fc2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.attn.proj(self.attn.qkv(inputs))
        return self.mlp.linear_fc2(torch.relu(self.mlp.linear_fc1(hidden)))


class TinyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(8, 16)
        self.linear_fc2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(torch.relu(self.linear_fc1(inputs)))


class TinyLanguageBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Namespace()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.self_attn.o_proj = nn.Linear(8, 8)
        self.mlp = Namespace()
        self.mlp.gate_proj = nn.Linear(8, 16)
        self.mlp.up_proj = nn.Linear(8, 16)
        self.mlp.down_proj = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.self_attn.o_proj(self.self_attn.q_proj(inputs))
        return self.mlp.down_proj(
            torch.relu(self.mlp.gate_proj(hidden)) * self.mlp.up_proj(hidden)
        )


def test_qwen3_vl_target_discovery_classifies_components() -> None:
    model = TinyQwen3VL()

    resolved, targets = get_target_linears(
        model, model_type="auto", include_lm_head=True
    )

    assert resolved == "qwen3_vl"
    assert sum(target.component == "language" for target in targets) == 5
    assert sum(target.component == "vision_encoder" for target in targets) == 4
    assert sum(target.component == "vision_merger" for target in targets) == 4
    assert sum(target.component == "lm_head" for target in targets) == 1


def test_qwen3_vl_can_patch_language_without_touching_vision() -> None:
    model = TinyQwen3VL()

    report = patch_causal_lm_linears(
        model,
        weight_format=get_fp4_format("mxfp4"),
        model_type="qwen3_vl",
        components=["language"],
    )

    assert report.replaced == 5
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, QuantLinear)
    assert isinstance(model.model.visual.blocks[0].attn.qkv, nn.Linear)
    assert isinstance(model.model.visual.merger.linear_fc1, nn.Linear)


def test_loaded_qwen3_vl_installs_weight_activation_quantization() -> None:
    model = TinyQwen3VL()
    original_weight = model.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()

    report = patch_loaded_qwen3_vl(
        model,
        weight_format="mxfp4",
        activation_format="nvfp4",
        components=["language"],
        prequantized_weight=True,
    )

    q_proj = model.model.language_model.layers[0].self_attn.q_proj
    assert report is not None and report.replaced == 5
    assert isinstance(q_proj, QuantLinear)
    assert q_proj.activation_format is not None
    assert q_proj.activation_format.name == "nvfp4"
    assert torch.equal(q_proj.weight, original_weight)


def test_multimodal_mapping_batch_collects_all_selected_components() -> None:
    model = TinyQwen3VL()
    _, targets = get_target_linears(
        model,
        model_type="qwen3_vl",
        include_lm_head=False,
        components=["language", "vision_encoder", "vision_merger"],
    )

    capture = collect_linear_calibration(
        model=model,
        targets=targets,
        batches=[
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.randn(4, 8),
            }
        ],
        max_tokens_per_layer=16,
        capture_mode="gram",
    )

    assert len(capture.grams) == len(targets)
    assert capture.token_counts["model.language_model.layers.0.self_attn.q_proj"] == 3
    assert capture.token_counts["model.visual.blocks.0.attn.qkv"] == 4


def test_qwen3_vl_moe_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="MoE is not supported"):
        get_target_linears(
            TinyQwen3VL("qwen3_vl_moe"), model_type="auto", include_lm_head=False
        )


def test_qwen3_vl_processor_batch_uses_structured_image_message() -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["type"] == "image"
            assert messages[0]["content"][1]["text"] == "What is shown?"
            assert kwargs["tokenize"] is True
            return {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "pixel_values": torch.randn(2, 8),
                "token_type_ids": torch.ones(1, 2, dtype=torch.long),
            }

    batch = build_qwen3_vl_calibration_batch(
        processor=Processor(), image=object(), prompt="What is shown?"
    )

    assert set(batch) == {"input_ids", "pixel_values"}
