from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Evaluate dense Qwen3-VL checkpoints with lmms-eval.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Base or SepQuant checkpoint path.")
    parser.add_argument("--tasks", nargs="+", default=["mme"])
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-path", type=Path, default=None)
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
    if isinstance(args.model_args, str):
        args.model_args = _parse_json_object(args.model_args, "--model-args")
    elif args.model_args is None:
        args.model_args = {}
    return args


def build_lmms_eval_command(args: argparse.Namespace) -> list[str]:
    model_args = {"pretrained": args.model, **args.model_args}
    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "qwen3_vl",
        "--model_args",
        _serialize_model_args(model_args),
        "--tasks",
        ",".join(args.tasks),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.output_path is not None:
        command.extend(["--output_path", str(args.output_path)])
    if args.log_samples:
        command.append("--log_samples")
    return command


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("lmms_eval") is None:
        raise SystemExit(
            "lmms-eval is required for multimodal evaluation. "
            "Install it with: pip install -e '.[multimodal]'"
        )
    subprocess.run(build_lmms_eval_command(args), check=True)


def _serialize_model_args(values: dict[str, Any]) -> str:
    serialized = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        if "," in str(value):
            raise ValueError(f"lmms-eval model arg {key!r} cannot contain a comma")
        serialized.append(f"{key}={value}")
    return ",".join(serialized)


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
