from __future__ import annotations

import json
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def save_experiment_run(
    *,
    experiment_dir: str | Path,
    run_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> Path:
    run_dir = _unique_run_dir(
        Path(experiment_dir),
        run_name=_build_run_name(run_type=run_type, config=config),
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_type": run_type,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "artifacts.json", artifacts or {})
    return run_dir


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _build_run_name(*, run_type: str, config: dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = Path(str(config.get("model", "model"))).name
    dataset = str(config.get("dataset", config.get("tasks", "data")))
    method = _method_name(config)
    parts = [timestamp, run_type, model, dataset, method]
    return "_".join(_sanitize(part) for part in parts if part)


def _method_name(config: dict[str, Any]) -> str:
    plan = config.get("quantization_plan")
    if plan:
        return Path(str(plan)).stem
    weight_format = config.get("weight_format", "none")
    activation_format = config.get("activation_format", "none")
    return f"w-{weight_format}_a-{activation_format}"


def _unique_run_dir(root: Path, *, run_name: str) -> Path:
    candidate = root / run_name
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = root / f"{run_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _sanitize(value: Any) -> str:
    text = str(value).strip().replace("/", "-").replace("\\", "-")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
