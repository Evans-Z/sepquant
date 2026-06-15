from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from sepquant.calibration import (
    build_calibration_batches,
    collect_linear_calibration,
    save_calibration_artifact,
)
from sepquant.models import get_target_linears
from sepquant.models.load import parse_dtype, resolve_device


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Collect reusable calibration activations for target linear layers.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen", "qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "gemma", "opt", "generic"],
    )
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-tokens-per-layer", type=int, default=2048)
    parser.add_argument("--capture-mode", choices=["inputs", "gram", "both"], default="gram")
    parser.add_argument("--input-layer-patterns", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if args.output_dir is None:
        parser.error("--output-dir is required unless provided by --config")
    if isinstance(args.output_dir, str):
        args.output_dir = Path(args.output_dir)
    if isinstance(args.input_layer_patterns, str):
        args.input_layer_patterns = [
            pattern.strip() for pattern in args.input_layer_patterns.split(",") if pattern.strip()
        ]
    return args


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=parse_dtype(args.dtype),
        device_map=args.device if args.device == "auto" else None,
        trust_remote_code=True,
    )
    if args.device != "auto":
        model.to(resolve_device(args.device))
    model.eval()

    resolved_model_type, targets = get_target_linears(
        model,
        model_type=args.model_type,
        include_lm_head=args.include_lm_head,
    )
    print(f"Found {len(targets)} target linear layers (model_type={resolved_model_type}).")

    batches = build_calibration_batches(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        nsamples=args.nsamples,
        seed=args.seed,
        sequence_length=args.sequence_length,
    )
    capture = collect_linear_calibration(
        model=model,
        targets=targets,
        batches=batches,
        max_tokens_per_layer=args.max_tokens_per_layer,
        capture_mode=args.capture_mode,
        input_layer_patterns=args.input_layer_patterns,
    )
    save_calibration_artifact(
        output_dir=args.output_dir,
        inputs=capture.inputs,
        grams=capture.grams,
        metadata={
            "model": args.model,
            "model_type": resolved_model_type,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "text_column": args.text_column,
            "nsamples": args.nsamples,
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "capture_mode": args.capture_mode,
            "input_layer_patterns": args.input_layer_patterns,
            "token_counts": capture.token_counts,
        },
    )
    print(f"Saved calibration artifact to {args.output_dir}")


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON config files are supported for now: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return config


if __name__ == "__main__":
    main()

