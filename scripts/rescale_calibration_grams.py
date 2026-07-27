#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

METADATA_FILENAME = "metadata.json"
NORMALIZATION_CHOICES = ("raw_sum", "token", "sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale saved calibration Gram matrices between raw-sum, token-normalized, "
            "and sample-normalized conventions."
        )
    )
    parser.add_argument("calibration_dir", type=Path, help="Input calibration artifact directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output calibration artifact directory. Required unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input calibration artifact in place.",
    )
    parser.add_argument(
        "--from-normalization",
        choices=NORMALIZATION_CHOICES,
        default="token",
        help="Current Gram normalization convention.",
    )
    parser.add_argument(
        "--to-normalization",
        choices=NORMALIZATION_CHOICES,
        default="sample",
        help="Target Gram normalization convention.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned scale factors without writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.calibration_dir
    output_dir = args.output_dir

    if args.in_place and output_dir is not None:
        raise ValueError("Use either --in-place or --output-dir, not both")
    if not args.in_place and output_dir is None:
        raise ValueError("--output-dir is required unless --in-place is set")
    if args.from_normalization == args.to_normalization:
        raise ValueError("--from-normalization and --to-normalization are identical")

    metadata = _load_metadata(input_dir)
    layer_factors = _layer_scale_factors(
        metadata,
        from_normalization=args.from_normalization,
        to_normalization=args.to_normalization,
    )

    if args.dry_run:
        _print_summary(layer_factors, dry_run=True)
        return

    work_dir = input_dir if args.in_place else _copy_artifact(input_dir, output_dir, overwrite=args.overwrite)
    metadata = _load_metadata(work_dir)

    updated = 0
    for layer_name, factor in layer_factors.items():
        layer_metadata = metadata["layers"][layer_name]
        gram_file = layer_metadata.get("gram_file")
        if gram_file is None:
            continue

        gram_path = work_dir / gram_file
        gram = torch.load(gram_path, map_location="cpu")
        torch.save(gram * factor, gram_path)
        layer_metadata["gram_rescale_factor"] = factor
        updated += 1

    metadata["gram_normalization"] = args.to_normalization
    metadata["gram_rescale"] = {
        "from_normalization": args.from_normalization,
        "to_normalization": args.to_normalization,
        "updated_layers": updated,
        "scale_factor_min": min(layer_factors.values()) if layer_factors else None,
        "scale_factor_max": max(layer_factors.values()) if layer_factors else None,
    }
    _save_metadata(work_dir, metadata)
    _print_summary(layer_factors, dry_run=False)
    print(f"Saved rescaled calibration artifact to {work_dir}")


def _load_metadata(calibration_dir: Path) -> dict[str, Any]:
    metadata_path = calibration_dir / METADATA_FILENAME
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid calibration metadata: {metadata_path}")
    return metadata


def _save_metadata(calibration_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = calibration_dir / METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _copy_artifact(input_dir: Path, output_dir: Path | None, *, overwrite: bool) -> Path:
    if output_dir is None:
        raise ValueError("--output-dir is required unless --in-place is set")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    return output_dir


def _layer_scale_factors(
    metadata: dict[str, Any],
    *,
    from_normalization: str,
    to_normalization: str,
) -> dict[str, float]:
    layers = metadata.get("layers", {})
    token_counts = metadata.get("token_counts", {})
    nsamples = metadata.get("nsamples")

    if not isinstance(layers, dict):
        raise ValueError("metadata.json does not contain a valid layers object")
    if not isinstance(token_counts, dict):
        raise ValueError("metadata.json does not contain a valid token_counts object")
    if not isinstance(nsamples, int) or nsamples <= 0:
        raise ValueError("metadata.json does not contain a positive integer nsamples")

    factors = {}
    for layer_name, layer_metadata in layers.items():
        if not isinstance(layer_metadata, dict) or layer_metadata.get("gram_file") is None:
            continue
        from_denom = _normalization_denominator(
            from_normalization,
            layer_name=layer_name,
            token_counts=token_counts,
            nsamples=nsamples,
        )
        to_denom = _normalization_denominator(
            to_normalization,
            layer_name=layer_name,
            token_counts=token_counts,
            nsamples=nsamples,
        )
        factors[layer_name] = from_denom / to_denom
    return factors


def _normalization_denominator(
    normalization: str,
    *,
    layer_name: str,
    token_counts: dict[str, Any],
    nsamples: int,
) -> float:
    if normalization == "raw_sum":
        return 1.0
    if normalization == "sample":
        return float(nsamples)
    if normalization == "token":
        token_count = token_counts.get(layer_name)
        if not isinstance(token_count, int) or token_count <= 0:
            raise ValueError(f"Missing positive token count for layer {layer_name!r}")
        return float(token_count)
    raise ValueError(f"Unsupported normalization: {normalization}")


def _print_summary(layer_factors: dict[str, float], *, dry_run: bool) -> None:
    prefix = "Would rescale" if dry_run else "Rescaled"
    if not layer_factors:
        print(f"{prefix} 0 Gram matrices.")
        return
    factors = list(layer_factors.values())
    print(
        f"{prefix} {len(factors)} Gram matrices "
        f"(scale min={min(factors):.6g}, max={max(factors):.6g})."
    )


if __name__ == "__main__":
    main()
