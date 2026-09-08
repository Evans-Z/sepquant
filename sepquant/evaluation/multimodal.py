from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from sepquant.experiments import save_experiment_run, write_json
from sepquant.models import patch_loaded_qwen3_vl


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Evaluate dense Qwen3-VL checkpoints with lmms-eval.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Base Qwen3-VL model id or path.")
    parser.add_argument("--model-type", default="qwen3_vl", choices=["qwen3_vl"])
    parser.add_argument(
        "--pre-quant-model",
        "--pre-quantized-model",
        dest="pre_quant_model",
        default=None,
        help="Saved SepQuant checkpoint evaluated instead of the base model.",
    )
    parser.add_argument("--weight-format", default="none")
    parser.add_argument("--activation-format", default="none")
    parser.add_argument("--quantization-plan", default=None)
    parser.add_argument("--rotation", default="none", choices=["none", "block_hadamard"])
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["language"],
        choices=["language", "vision_merger", "vision_encoder", "lm_head"],
    )
    parser.add_argument("--tasks", nargs="+", default=["mme"])
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument(
        "--model-args",
        default=None,
        help="Additional lmms-eval model args as a JSON object.",
    )
    parser.add_argument("--log-samples", action="store_true")
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if isinstance(args.tasks, str):
        args.tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if isinstance(args.output_path, str):
        args.output_path = Path(args.output_path)
    if isinstance(args.experiment_dir, str):
        args.experiment_dir = Path(args.experiment_dir)
    if isinstance(args.model_args, str):
        args.model_args = _parse_json_object(args.model_args, "--model-args")
    elif args.model_args is None:
        args.model_args = {}
    if isinstance(args.components, str):
        args.components = [item.strip() for item in args.components.split(",") if item.strip()]
    return args


def build_lmms_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_source = args.pre_quant_model or args.model
    kwargs = {
        "pretrained": model_source,
        "batch_size": args.batch_size,
        **args.model_args,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("lmms_eval") is None:
        raise SystemExit(
            "lmms-eval is required for multimodal evaluation. "
            "Install it with: pip install -e '.[multimodal]'"
        )
    simple_evaluate, model_cls = _import_lmms_eval()
    results, patch_report = evaluate_multimodal(
        args,
        simple_evaluate=simple_evaluate,
        model_cls=model_cls,
    )
    if patch_report is not None:
        print(
            f"Installed SepQuant W/A modules in {patch_report.replaced} Qwen3-VL layers "
            f"(W={args.weight_format}, A={args.activation_format})."
        )
    if args.output_path is not None:
        write_json(args.output_path, results)
        print(f"Saved multimodal evaluation results to {args.output_path}")
    if args.experiment_dir is not None:
        run_dir = save_experiment_run(
            experiment_dir=args.experiment_dir,
            run_type="multimodal_tasks",
            config=vars(args),
            metrics={
                "tasks": args.tasks,
                "weight_format": args.weight_format,
                "activation_format": args.activation_format,
                "results": results.get("results", {}),
            },
            artifacts={
                "quantization_plan": args.quantization_plan,
                "pre_quant_model": args.pre_quant_model,
                "output_path": args.output_path,
            },
        )
        print(f"Saved experiment run to {run_dir}")


def evaluate_multimodal(args: argparse.Namespace, *, simple_evaluate, model_cls):
    model_kwargs = _filter_supported_init_kwargs(model_cls, build_lmms_model_kwargs(args))
    lm = model_cls(**model_kwargs)
    if args.activation_format != "none" and getattr(lm, "world_size", 1) != 1:
        raise RuntimeError(
            "SepQuant activation patching currently requires a single lmms-eval process; "
            "single-process device_map sharding remains supported"
        )

    plan = _resolve_quantization_plan(args)
    patch_report = patch_loaded_qwen3_vl(
        lm.model,
        weight_format=args.weight_format,
        activation_format=args.activation_format,
        components=args.components,
        include_lm_head=args.include_lm_head,
        quantization_plan=plan,
        prequantized_weight=args.pre_quant_model is not None,
        rotation=args.rotation,
    )

    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
    )
    return results, patch_report


def _import_lmms_eval():
    from lmms_eval.evaluator import simple_evaluate

    try:
        from lmms_eval.models.chat.qwen3_vl import Qwen3_VL
    except ImportError:
        from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
    return simple_evaluate, Qwen3_VL


def _filter_supported_init_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    supported = set(signature.parameters) - {"self", "kwargs"}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"Warning: current lmms-eval Qwen3-VL adapter ignored model args: {dropped}")
    return filtered


def _resolve_quantization_plan(args: argparse.Namespace) -> str | None:
    if args.quantization_plan is not None:
        return args.quantization_plan
    if args.pre_quant_model is None:
        return None
    path = Path(args.pre_quant_model) / "quantization_plan.json"
    return str(path) if path.exists() else None


def _parse_json_object(value: str, arg_name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return parsed


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
