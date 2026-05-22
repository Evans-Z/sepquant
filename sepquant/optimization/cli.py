from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from sepquant.models import patch_causal_lm_linears
from sepquant.models.load import parse_dtype, resolve_device
from sepquant.models.patch import get_target_linears
from sepquant.optimization.layerwise import build_plan_from_results, optimize_layers
from sepquant.optimization.methods import build_layer_optimizer
from sepquant.quantization import QuantizationPlan


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Run layer-wise quantization optimization from saved calibration artifacts.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen", "qwen2", "qwen3", "llama", "mistral", "gemma", "opt", "generic"],
    )
    parser.add_argument("--method", default="weight_format_search")
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--candidates", nargs="+", default=["mxfp4", "nvfp4"])
    parser.add_argument("--weight-format", default="mxfp4", choices=["mxfp4", "nvfp4"])
    parser.add_argument(
        "--activation-format",
        default="none",
        choices=["none", "mxfp4", "mxfp4_search", "nvfp4", "nvfp4_search"],
    )
    parser.add_argument("--gptq-damp-percent", type=float, default=0.01)
    parser.add_argument("--mxfp4-scale-offsets", nargs="+", type=int, default=[-2, -1, 0, 1, 2])
    parser.add_argument("--mxfp4-scale-objective", default="block", choices=["identity", "diag", "block"])
    parser.add_argument(
        "--nvfp4-scale-code-offsets",
        nargs="+",
        type=int,
        default=[-3, -2, -1, 0, 1, 2, 3],
    )
    parser.add_argument("--rotation", default="none", choices=["none", "block_hadamard"])
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output-plan-path", type=Path, default=None)
    parser.add_argument("--save-quantized-checkpoint", type=Path, default=None)
    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if args.calibration_dir is None:
        parser.error("--calibration-dir is required unless provided by --config")
    if args.output_plan_path is None:
        parser.error("--output-plan-path is required unless provided by --config")
    if isinstance(args.calibration_dir, str):
        args.calibration_dir = Path(args.calibration_dir)
    if isinstance(args.output_plan_path, str):
        args.output_plan_path = Path(args.output_plan_path)
    if isinstance(args.save_quantized_checkpoint, str):
        args.save_quantized_checkpoint = Path(args.save_quantized_checkpoint)
    if isinstance(args.candidates, str):
        args.candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    if isinstance(args.mxfp4_scale_offsets, str):
        args.mxfp4_scale_offsets = [
            int(item.strip()) for item in args.mxfp4_scale_offsets.split(",") if item.strip()
        ]
    if isinstance(args.nvfp4_scale_code_offsets, str):
        args.nvfp4_scale_code_offsets = [
            int(item.strip()) for item in args.nvfp4_scale_code_offsets.split(",") if item.strip()
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

    optimizer = build_layer_optimizer(
        method=args.method,
        candidates=args.candidates,
        activation_format=args.activation_format,
        weight_format=args.weight_format,
        gptq_damp_percent=args.gptq_damp_percent,
        mxfp4_scale_offsets=args.mxfp4_scale_offsets,
        mxfp4_scale_objective=args.mxfp4_scale_objective,
        nvfp4_scale_code_offsets=args.nvfp4_scale_code_offsets,
        rotation=args.rotation,
        device=args.device,
    )
    results = optimize_layers(
        targets=targets,
        calibration_dir=args.calibration_dir,
        optimizer=optimizer,
    )
    plan = build_plan_from_results(
        results=results,
        metadata={
            "method": args.method,
            "model": args.model,
            "model_type": resolved_model_type,
            "calibration_dir": str(args.calibration_dir),
            "candidates": args.candidates,
            "weight_format": args.weight_format,
            "activation_format": args.activation_format,
            "gptq_damp_percent": args.gptq_damp_percent,
            "mxfp4_scale_offsets": args.mxfp4_scale_offsets,
            "mxfp4_scale_objective": args.mxfp4_scale_objective,
            "nvfp4_scale_code_offsets": args.nvfp4_scale_code_offsets,
            "rotation": args.rotation,
            "device": args.device,
        },
    )

    args.output_plan_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    print(f"Saved quantization plan to {args.output_plan_path}")

    if args.save_quantized_checkpoint is not None:
        _save_sepquant_checkpoint(
            model=model,
            tokenizer=tokenizer,
            model_type=resolved_model_type,
            plan=plan,
            results=results,
            checkpoint_dir=args.save_quantized_checkpoint,
            include_lm_head=args.include_lm_head,
        )


def _save_sepquant_checkpoint(
    *,
    model,
    tokenizer,
    model_type: str,
    plan: dict[str, Any],
    results,
    checkpoint_dir: Path,
    include_lm_head: bool,
) -> None:
    quantization_plan = QuantizationPlan.from_dict(plan)
    patch_report = patch_causal_lm_linears(
        model,
        weight_format=None,
        activation_format=None,
        model_type=model_type,
        include_lm_head=include_lm_head,
        quantization_plan=quantization_plan,
        rotation=plan.get("metadata", {}).get("rotation", "none"),
    )
    optimized_weights = {
        result.layer_name: result.optimized_weight
        for result in results
        if result.optimized_weight is not None
    }
    for module_name, module in model.named_modules():
        if module_name not in optimized_weights:
            continue
        if hasattr(module, "weight"):
            module.weight.copy_(optimized_weights[module_name].to(module.weight.device, module.weight.dtype))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)
    model.save_pretrained(checkpoint_dir, safe_serialization=False)
    with (checkpoint_dir / "quantization_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "transformers_pretrained",
                "model_type": model_type,
                "rotation": _checkpoint_rotation(plan),
                "patched_layers": patch_report.replaced,
            },
            handle,
            indent=2,
        )
    print(f"Saved SepQuant checkpoint to {checkpoint_dir}")


def _checkpoint_rotation(plan: dict[str, Any]) -> str:
    layers = plan.get("layers", {})
    if not isinstance(layers, dict) or not layers:
        return plan.get("metadata", {}).get("rotation", "none")

    rotations = {
        spec.get("rotation", "none")
        for spec in layers.values()
        if isinstance(spec, dict) and spec.get("enabled", True)
    }
    if len(rotations) == 1:
        return next(iter(rotations))
    return "none"


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

