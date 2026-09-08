import argparse

from torch import nn

from sepquant.evaluation.multimodal import build_lmms_model_kwargs, evaluate_multimodal
from sepquant.models import QuantLinear


def test_lmms_model_kwargs_target_qwen3_vl_checkpoint() -> None:
    args = argparse.Namespace(
        model="outputs/checkpoints/qwen3-vl",
        pre_quant_model="outputs/checkpoints/qwen3-vl-mxfp4",
        tasks=["mme", "mmmu_val"],
        num_fewshot=0,
        batch_size="1",
        limit=8,
        device="cuda:0",
        output_path=None,
        model_args={"max_pixels": 1024},
        dtype="bfloat16",
        log_samples=False,
    )

    model_kwargs = build_lmms_model_kwargs(args)

    assert model_kwargs == {
        "pretrained": "outputs/checkpoints/qwen3-vl-mxfp4",
        "batch_size": "1",
        "device": "cuda:0",
        "device_map": "cuda:0",
        "max_pixels": 1024,
    }


def test_lmms_model_kwargs_preserve_explicit_device_map() -> None:
    args = argparse.Namespace(
        model="Qwen/Qwen3-VL-4B-Instruct",
        pre_quant_model=None,
        batch_size="1",
        device="cuda:0",
        model_args={"device_map": "auto"},
    )

    model_kwargs = build_lmms_model_kwargs(args)

    assert model_kwargs["device"] == "cuda:0"
    assert model_kwargs["device_map"] == "auto"


def test_in_process_eval_keeps_activation_quantization_installed() -> None:
    class FakeHFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"model_type": "qwen3_vl"})()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.q_proj = nn.Linear(8, 8)

    class FakeLmmsModel:
        def __init__(self, pretrained, batch_size, device) -> None:
            self.loaded_from = pretrained
            self.batch_size = batch_size
            self.device = device
            self.world_size = 1
            self.model = FakeHFModel()

    def fake_simple_evaluate(*, model, **kwargs):
        q_proj = model.model.model.language_model.q_proj
        assert isinstance(q_proj, QuantLinear)
        assert q_proj.activation_format is not None
        assert q_proj.activation_format.name == "nvfp4"
        return {"results": {"mme": {"score": 1.0}}}

    args = argparse.Namespace(
        model="Qwen/Qwen3-VL-4B-Instruct",
        pre_quant_model=None,
        model_args={},
        batch_size="1",
        device="cuda:0",
        weight_format="mxfp4",
        activation_format="nvfp4",
        components=["language"],
        include_lm_head=False,
        quantization_plan=None,
        rotation="none",
        tasks=["mme"],
        num_fewshot=0,
        limit=1,
        log_samples=False,
    )

    results, report = evaluate_multimodal(
        args,
        simple_evaluate=fake_simple_evaluate,
        model_cls=FakeLmmsModel,
    )

    assert report is not None and report.replaced == 1
    assert results["results"]["mme"]["score"] == 1.0
