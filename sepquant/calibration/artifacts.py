from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import torch


INPUTS_DIRNAME = "inputs"
GRAMS_DIRNAME = "grams"
METADATA_FILENAME = "metadata.json"


def save_calibration_artifact(
    *,
    output_dir: str | Path,
    activations: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, Any],
    inputs: dict[str, torch.Tensor] | None = None,
    grams: dict[str, torch.Tensor] | None = None,
) -> None:
    root = Path(output_dir)
    if activations is not None:
        if inputs is not None:
            raise ValueError("Use either 'activations' or 'inputs', not both")
        inputs = activations
    inputs = inputs or {}
    grams = grams or {}

    input_files = _save_tensor_group(root=root, dirname=INPUTS_DIRNAME, tensors=inputs)
    gram_files = _save_tensor_group(root=root, dirname=GRAMS_DIRNAME, tensors=grams)

    layer_names = sorted(set(input_files) | set(gram_files))
    layers = {}
    for layer_name in layer_names:
        layer_metadata = {}
        if layer_name in input_files:
            layer_metadata["input_file"] = input_files[layer_name]
            layer_metadata["input_shape"] = list(inputs[layer_name].shape)
            layer_metadata["input_dtype"] = str(inputs[layer_name].dtype)
            # Backward-compatible fields for older input-only artifacts.
            layer_metadata["file"] = input_files[layer_name]
            layer_metadata["shape"] = list(inputs[layer_name].shape)
            layer_metadata["dtype"] = str(inputs[layer_name].dtype)
        if layer_name in gram_files:
            layer_metadata["gram_file"] = gram_files[layer_name]
            layer_metadata["gram_shape"] = list(grams[layer_name].shape)
            layer_metadata["gram_dtype"] = str(grams[layer_name].dtype)
        layers[layer_name] = layer_metadata

    metadata = dict(metadata)
    metadata["layers"] = layers
    with (root / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _save_tensor_group(
    *,
    root: Path,
    dirname: str,
    tensors: dict[str, torch.Tensor],
) -> dict[str, str]:
    if not tensors:
        return {}

    tensor_dir = root / dirname
    tensor_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for layer_name, tensor in tensors.items():
        filename = f"{_encode_layer_name(layer_name)}.pt"
        torch.save(tensor.cpu(), tensor_dir / filename)
        files[layer_name] = str(Path(dirname) / filename)
    return files


def load_calibration_artifact(
    calibration_dir: str | Path,
    *,
    layer_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    return load_calibration_inputs(calibration_dir, layer_names=layer_names)


def load_calibration_inputs(
    calibration_dir: str | Path,
    *,
    layer_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    return _load_tensor_group(
        calibration_dir,
        layer_names=layer_names,
        preferred_key="input_file",
        legacy_key="file",
    )


def load_calibration_grams(
    calibration_dir: str | Path,
    *,
    layer_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    return _load_tensor_group(
        calibration_dir,
        layer_names=layer_names,
        preferred_key="gram_file",
        legacy_key=None,
    )


def load_layer_input(calibration_dir: str | Path, layer_name: str) -> torch.Tensor | None:
    tensors = load_calibration_inputs(calibration_dir, layer_names=[layer_name])
    return tensors.get(layer_name)


def load_layer_gram(calibration_dir: str | Path, layer_name: str) -> torch.Tensor | None:
    tensors = load_calibration_grams(calibration_dir, layer_names=[layer_name])
    return tensors.get(layer_name)


def _load_tensor_group(
    calibration_dir: str | Path,
    *,
    layer_names: list[str] | None,
    preferred_key: str,
    legacy_key: str | None,
) -> dict[str, torch.Tensor]:
    root = Path(calibration_dir)
    metadata_path = root / METADATA_FILENAME
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    layers = metadata.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError(f"Invalid calibration metadata: {metadata_path}")

    selected_names = layer_names if layer_names is not None else list(layers)
    tensors = {}
    for layer_name in selected_names:
        if layer_name not in layers:
            continue
        layer_metadata = layers[layer_name]
        file_path = layer_metadata.get(preferred_key)
        if file_path is None and legacy_key is not None:
            file_path = layer_metadata.get(legacy_key)
        if file_path is None:
            continue
        tensor_path = root / file_path
        tensors[layer_name] = torch.load(tensor_path, map_location="cpu")
    return tensors


def load_calibration_metadata(calibration_dir: str | Path) -> dict[str, Any]:
    with (Path(calibration_dir) / METADATA_FILENAME).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid calibration metadata in {calibration_dir}")
    return metadata


def _encode_layer_name(layer_name: str) -> str:
    return quote(layer_name, safe="")


def _decode_layer_name(encoded: str) -> str:
    return unquote(encoded)

