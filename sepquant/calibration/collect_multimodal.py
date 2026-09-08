from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sepquant.calibration import (
    build_multimodal_calibration_batches,
    collect_linear_calibration,
    save_calibration_artifact,
)
from sepquant.models import get_target_linears, load_quantized_qwen3_vl


COMPONENT_CHOICES = ["language", "vision_merger", "vision_encoder"]


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Collect image-text calibration statistics for dense Qwen3-VL.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens-per-layer", type=int, default=2048)
    parser.add_argument("--capture-mode", choices=["inputs", "gram", "both"], default="gram")
    parser.add_argument("--input-layer-patterns", nargs="*", default=None)
    parser.add_argument("--components", nargs="+", choices=COMPONENT_CHOICES, default=["language"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if args.dataset is None:
        parser.error("--dataset is required unless provided by --config")
    if args.output_dir is None:
        parser.error("--output-dir is required unless provided by --config")
    args.output_dir = Path(args.output_dir)
    if isinstance(args.components, str):
        args.components = [item.strip() for item in args.components.split(",") if item.strip()]
    if isinstance(args.input_layer_patterns, str):
        args.input_layer_patterns = [
            item.strip() for item in args.input_layer_patterns.split(",") if item.strip()
        ]
    return args


def main() -> None:
    args = parse_args()
    loaded = load_quantized_qwen3_vl(
        model_name_or_path=args.model,
        weight_format="none",
        activation_format="none",
        device=args.device,
        dtype=args.dtype,
        components=args.components,
    )
    _, targets = get_target_linears(
        loaded.model,
        model_type="qwen3_vl",
        include_lm_head=False,
        components=args.components,
    )
    component_counts = Counter(target.component for target in targets)
    print(f"Found {len(targets)} Qwen3-VL targets: {dict(sorted(component_counts.items()))}")

    batches = build_multimodal_calibration_batches(
        processor=loaded.processor,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        image_column=args.image_column,
        text_column=args.text_column,
        prompt=args.prompt,
        nsamples=args.nsamples,
        seed=args.seed,
    )
    capture = collect_linear_calibration(
        model=loaded.model,
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
            "model_type": "qwen3_vl",
            "modality": "image_text",
            "components": args.components,
            "component_counts": dict(sorted(component_counts.items())),
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "image_column": args.image_column,
            "text_column": args.text_column,
            "prompt": args.prompt,
            "nsamples": args.nsamples,
            "seed": args.seed,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "capture_mode": args.capture_mode,
            "input_layer_patterns": args.input_layer_patterns,
            "token_counts": capture.token_counts,
        },
    )
    print(f"Saved multimodal calibration artifact to {args.output_dir}")


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
