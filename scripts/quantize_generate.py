#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sepquant.formats import get_fp4_format
from sepquant.models import patch_causal_lm_linears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fake-quantized generation smoke test.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen", "qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "gemma", "opt", "generic"],
    )
    parser.add_argument("--weight-format", default="mxfp4", choices=["none", "mxfp4", "nvfp4", "hif4"])
    parser.add_argument(
        "--activation-format",
        default="none",
        choices=["none", "mxfp4", "mxfp4_search", "nvfp4", "nvfp4_search", "hif4", "hif4_search"],
    )
    parser.add_argument("--rotation", default="none", choices=["none", "block_hadamard"])
    parser.add_argument("--device", default="auto", help="Device map value, or a torch device like cuda:0/cpu/mps.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--prompt", default="Explain the idea of decoupled quantization in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _parse_dtype(args.dtype)
    device_map = args.device if args.device == "auto" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model.to(_resolve_device(args.device))
    model.eval()

    if args.weight_format != "none":
        report = patch_causal_lm_linears(
            model,
            weight_format=get_fp4_format(args.weight_format),
            activation_format=(
                None if args.activation_format == "none" else get_fp4_format(args.activation_format)
            ),
            model_type=args.model_type,
            include_lm_head=args.include_lm_head,
            rotation=args.rotation,
        )
        print(
            f"Replaced {report.replaced} linear layers with QuantLinear "
            f"(model_type={report.model_type})."
        )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def _parse_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "mps":
        return torch.device("mps")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return torch.device(device)
    return torch.device(device)


if __name__ == "__main__":
    main()

