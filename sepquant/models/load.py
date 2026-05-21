from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from sepquant.formats import get_fp4_format
from sepquant.models.patch import PatchReport, patch_causal_lm_linears
from sepquant.quantization import QuantizationPlan


@dataclass(frozen=True)
class LoadedCausalLM:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    patch_report: PatchReport | None


def load_quantized_causal_lm(
    *,
    model_name_or_path: str,
    model_type: str,
    weight_format: str,
    activation_format: str,
    device: str,
    dtype: str,
    include_lm_head: bool,
    quantization_plan: str | None = None,
    pre_quant_model: str | None = None,
) -> LoadedCausalLM:
    torch_dtype = parse_dtype(dtype)
    device_map = device if device == "auto" else None
    model_source = pre_quant_model or model_name_or_path
    tokenizer_source = _resolve_tokenizer_source(
        model_name_or_path=model_name_or_path,
        pre_quant_model=pre_quant_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model.to(resolve_device(device))
    model.eval()

    plan = QuantizationPlan.from_file(quantization_plan) if quantization_plan else None

    patch_report = None
    if pre_quant_model is not None:
        if activation_format != "none":
            if weight_format == "none":
                raise ValueError("weight_format must describe pre_quant_model weights when activation_format is enabled")
            patch_report = patch_causal_lm_linears(
                model,
                weight_format=get_fp4_format(weight_format),
                activation_format=get_fp4_format(activation_format),
                model_type=model_type,
                include_lm_head=include_lm_head,
                quantization_plan=None,
                prequantized_weight=True,
            )
        return LoadedCausalLM(model=model, tokenizer=tokenizer, patch_report=patch_report)

    if weight_format != "none" or plan is not None:
        patch_report = patch_causal_lm_linears(
            model,
            weight_format=None if weight_format == "none" else get_fp4_format(weight_format),
            activation_format=None if activation_format == "none" else get_fp4_format(activation_format),
            model_type=model_type,
            include_lm_head=include_lm_head,
            quantization_plan=plan,
        )

    return LoadedCausalLM(model=model, tokenizer=tokenizer, patch_report=patch_report)


def _resolve_tokenizer_source(*, model_name_or_path: str, pre_quant_model: str | None) -> str:
    if pre_quant_model is None:
        return model_name_or_path

    checkpoint = Path(pre_quant_model)
    if not checkpoint.exists():
        return pre_quant_model

    tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    }
    if any((checkpoint / filename).exists() for filename in tokenizer_files):
        return pre_quant_model
    return model_name_or_path


def parse_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized in {"mps", "apple", "apple-silicon", "apple_silicon"}:
        return torch.device("mps")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return torch.device(device)
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

