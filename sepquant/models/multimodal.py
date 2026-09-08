from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoProcessor, PreTrainedModel

from sepquant.formats import get_fp4_format
from sepquant.models.load import parse_dtype, resolve_device
from sepquant.models.patch import ModelComponent, PatchReport, patch_causal_lm_linears
from sepquant.quantization import QuantizationPlan


DEFAULT_QWEN3_VL_COMPONENTS: tuple[ModelComponent, ...] = ("language",)


@dataclass(frozen=True)
class LoadedMultimodalModel:
    model: PreTrainedModel
    processor: Any
    patch_report: PatchReport | None


def load_quantized_qwen3_vl(
    *,
    model_name_or_path: str,
    weight_format: str,
    activation_format: str,
    device: str,
    dtype: str,
    components: tuple[ModelComponent, ...] | list[ModelComponent] = DEFAULT_QWEN3_VL_COMPONENTS,
    include_lm_head: bool = False,
    quantization_plan: str | None = None,
    pre_quant_model: str | None = None,
    rotation: str = "none",
) -> LoadedMultimodalModel:
    """Load and optionally fake-quantize a dense Qwen3-VL checkpoint."""

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError(
            "Qwen3-VL requires a Transformers release that provides "
            "Qwen3VLForConditionalGeneration"
        )

    model_source = pre_quant_model or model_name_or_path
    processor_source = _resolve_processor_source(
        model_name_or_path=model_name_or_path,
        pre_quant_model=pre_quant_model,
    )
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    model = model_cls.from_pretrained(
        model_source,
        torch_dtype=parse_dtype(dtype),
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    _ensure_dense_qwen3_vl(model)
    if device != "auto":
        model.to(resolve_device(device))
    model.eval()

    plan = _resolve_plan(quantization_plan=quantization_plan, pre_quant_model=pre_quant_model)
    patch_report = patch_loaded_qwen3_vl(
        model,
        weight_format=weight_format,
        activation_format=activation_format,
        components=components,
        include_lm_head=include_lm_head,
        quantization_plan=plan,
        prequantized_weight=pre_quant_model is not None,
        rotation=rotation,
    )

    return LoadedMultimodalModel(model=model, processor=processor, patch_report=patch_report)


def patch_loaded_qwen3_vl(
    model: PreTrainedModel,
    *,
    weight_format: str,
    activation_format: str,
    components: tuple[ModelComponent, ...] | list[ModelComponent] = DEFAULT_QWEN3_VL_COMPONENTS,
    include_lm_head: bool = False,
    quantization_plan: str | Path | QuantizationPlan | None = None,
    prequantized_weight: bool = False,
    rotation: str = "none",
) -> PatchReport | None:
    """Install SepQuant W/A modules into an already loaded dense Qwen3-VL model."""

    _ensure_dense_qwen3_vl(model)
    if isinstance(quantization_plan, QuantizationPlan):
        plan = quantization_plan
    elif quantization_plan is not None:
        plan = QuantizationPlan.from_file(quantization_plan)
    else:
        plan = None

    should_patch = (
        weight_format != "none"
        or activation_format != "none"
        or plan is not None
        or rotation != "none"
    )
    if not should_patch:
        return None
    if weight_format == "none" and plan is None:
        raise ValueError(
            "weight_format or quantization_plan is required when activation quantization "
            "or rotation is enabled"
        )

    return patch_causal_lm_linears(
        model,
        weight_format=None if weight_format == "none" else get_fp4_format(weight_format),
        activation_format=(
            None if activation_format == "none" else get_fp4_format(activation_format)
        ),
        model_type="qwen3_vl",
        include_lm_head=include_lm_head,
        quantization_plan=plan,
        prequantized_weight=prequantized_weight,
        rotation=rotation,
        override_plan_activation_format=activation_format != "none",
        components=components,
    )


def _ensure_dense_qwen3_vl(model: PreTrainedModel) -> None:
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type == "qwen3_vl_moe":
        raise ValueError("Qwen3-VL MoE is not supported yet; use a dense Qwen3-VL checkpoint")
    if model_type != "qwen3_vl":
        raise ValueError(f"Expected a dense Qwen3-VL checkpoint, got model_type={model_type!r}")


def _resolve_plan(
    *, quantization_plan: str | None, pre_quant_model: str | None
) -> QuantizationPlan | None:
    if quantization_plan is not None:
        return QuantizationPlan.from_file(quantization_plan)
    if pre_quant_model is None:
        return None
    path = Path(pre_quant_model) / "quantization_plan.json"
    return QuantizationPlan.from_file(path) if path.exists() else None


def _resolve_processor_source(*, model_name_or_path: str, pre_quant_model: str | None) -> str:
    if pre_quant_model is None:
        return model_name_or_path
    checkpoint = Path(pre_quant_model)
    if checkpoint.exists() and (checkpoint / "processor_config.json").exists():
        return pre_quant_model
    if checkpoint.exists() and (checkpoint / "preprocessor_config.json").exists():
        return pre_quant_model
    return model_name_or_path
