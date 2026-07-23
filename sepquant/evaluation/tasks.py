from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from sepquant.experiments import save_experiment_run, write_json
from sepquant.models import load_quantized_causal_lm
from sepquant.models.load import resolve_device


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Evaluate downstream tasks with lm-eval-harness and optional FP4 fake quantization.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen", "qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "gemma", "opt", "generic"],
    )
    parser.add_argument("--tasks", nargs="+", default=["hellaswag"])
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--weight-format", default="mxfp4", choices=["none", "mxfp4", "mxfp4_plus", "nvfp4", "hif4"])
    parser.add_argument(
        "--activation-format",
        default="none",
        choices=[
            "none",
            "mxfp4",
            "mxfp4_plus",
            "mxfp4_plus_search",
            "mxfp4_search",
            "nvfp4",
            "nvfp4_search",
            "hif4",
            "hif4_search",
        ],
    )
    parser.add_argument("--quantization-plan", default=None)
    parser.add_argument("--pre-quant-model", "--pre-quantized-model", dest="pre_quant_model", default=None)
    parser.add_argument("--rotation", default="none", choices=["none", "block_hadamard"])
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument(
        "--apply-chat-template",
        nargs="?",
        const=True,
        default=False,
        help="Apply the tokenizer chat template before evaluation. Optionally pass a template name/string.",
    )
    parser.add_argument("--system-instruction", default=None)
    parser.add_argument("--fewshot-as-multiturn", action="store_true")
    parser.add_argument(
        "--gen-kwargs",
        default=None,
        help='JSON object passed to lm-eval generation, e.g. \'{"max_gen_toks": 1024}\'.',
    )
    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if isinstance(args.tasks, str):
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if isinstance(args.output_path, str):
        args.output_path = Path(args.output_path)
    if isinstance(args.experiment_dir, str):
        args.experiment_dir = Path(args.experiment_dir)
    if isinstance(args.gen_kwargs, str):
        args.gen_kwargs = _parse_json_object(args.gen_kwargs, "--gen-kwargs")
    return args


def main() -> None:
    args = parse_args()
    evaluator, HFLM = _import_lm_eval()

    loaded = load_quantized_causal_lm(
        model_name_or_path=args.model,
        model_type=args.model_type,
        weight_format=args.weight_format,
        activation_format=args.activation_format,
        device=args.device,
        dtype=args.dtype,
        include_lm_head=args.include_lm_head,
        quantization_plan=args.quantization_plan,
        pre_quant_model=args.pre_quant_model,
        rotation=args.rotation,
    )
    if args.pre_quant_model is not None:
        print(f"Loaded pre-quantized model from {args.pre_quant_model}; skipped weight quantization.")
    if loaded.patch_report is not None:
        print(
            f"Replaced {loaded.patch_report.replaced} linear layers with QuantLinear "
            f"(model_type={loaded.patch_report.model_type})."
        )

    hflm_cls = HFLM
    hflm_kwargs = _filter_supported_init_kwargs(
        hflm_cls,
        {
            "pretrained": loaded.model,
            "tokenizer": loaded.tokenizer,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": _lm_eval_device(args.device),
        },
    )
    lm = hflm_cls(**hflm_kwargs)
    evaluate_kwargs = {
        "model": lm,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "log_samples": args.log_samples,
    }
    if args.gen_kwargs is not None:
        evaluate_kwargs["gen_kwargs"] = args.gen_kwargs
    if args.apply_chat_template:
        evaluate_kwargs["apply_chat_template"] = args.apply_chat_template
    if args.system_instruction is not None:
        evaluate_kwargs["system_instruction"] = args.system_instruction
    if args.fewshot_as_multiturn:
        evaluate_kwargs["fewshot_as_multiturn"] = True
    evaluate_kwargs = _filter_supported_call_kwargs(evaluator.simple_evaluate, evaluate_kwargs)
    results = evaluator.simple_evaluate(**evaluate_kwargs)

    if args.output_path is not None:
        write_json(args.output_path, results)
        print(f"Saved results to {args.output_path}")

    compact_results = _compact_results(results)
    print(json.dumps(compact_results, indent=2, default=_json_default))
    if args.experiment_dir is not None:
        run_dir = save_experiment_run(
            experiment_dir=args.experiment_dir,
            run_type="tasks",
            config=vars(args),
            metrics={
                "task": "lm_eval",
                "tasks": args.tasks,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "weight_source": args.pre_quant_model or args.model,
                "base_model": args.model,
                "pre_quant_model": args.pre_quant_model,
                "quantization_mode": "pre_quantized" if args.pre_quant_model is not None else "runtime",
                "results": compact_results,
            },
            artifacts={
                "quantization_plan": args.quantization_plan,
                "pre_quant_model": args.pre_quant_model,
                "output_path": args.output_path,
            },
        )
        print(f"Saved experiment run to {run_dir}")


def _import_lm_eval():
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise SystemExit(
            "lm-eval-harness is required for downstream tasks. "
            "Install it with: pip install -e '.[eval]'"
        ) from exc
    return evaluator, HFLM


def _lm_eval_device(device: str) -> str:
    return str(resolve_device(device))


def _filter_supported_init_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}

    supported = set(signature.parameters) - {"self"}
    filtered = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in supported
    }
    dropped = sorted(key for key, value in kwargs.items() if value is not None and key not in supported)
    if dropped:
        print(f"Warning: current HFLM does not support these init kwargs and they were ignored: {dropped}")
    return filtered


def _filter_supported_call_kwargs(function, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}

    supported = set(signature.parameters)
    filtered = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in supported
    }
    dropped = sorted(key for key, value in kwargs.items() if value is not None and key not in supported)
    if dropped:
        print(f"Warning: current simple_evaluate does not support these kwargs and they were ignored: {dropped}")
    return filtered


def _compact_results(results: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for task, metrics in results.get("results", {}).items():
        compact[task] = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, int | float | str | bool)
        }
    return compact


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON config files are supported for now: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return config


def _parse_json_object(value: str, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be a JSON object: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object: {value}")
    return parsed


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()

