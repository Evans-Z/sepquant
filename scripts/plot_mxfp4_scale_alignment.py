#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch

from sepquant.formats.fp_ops import E2M1_MAX, quantize_e2m1, raw_e2m1_block_scale, to_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how maximum-absolute-value MXFP4 scaling can misalign FP4 "
            "levels with a real tensor block, compared with searched scale codes."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model id or local model path.")
    source.add_argument(
        "--tensor-path",
        type=Path,
        help="Path to a saved tensor or state dict (.pt/.pth) for offline plotting.",
    )
    parser.add_argument(
        "--tensor-name",
        help=(
            "Exact tensor name for a state dict or model. If omitted, the first "
            "2D weight tensor matching --tensor-regex is used."
        ),
    )
    parser.add_argument(
        "--tensor-regex",
        default=r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|fc|linear).*weight$",
        help="Regex used to select a tensor when --tensor-name is omitted.",
    )
    parser.add_argument("--block-size", type=int, default=32, help="MXFP4 block size.")
    parser.add_argument(
        "--search-offsets",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 0, 1, 2],
        help="Integer E8M0 exponent offsets searched around the default scale.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=200_000,
        help="Maximum number of blocks sampled when automatically selecting a block.",
    )
    parser.add_argument("--block-index", type=int, help="Use this flattened block index.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/mxfp4_scale_alignment.pdf"),
        help=(
            "Output path prefix. The script writes three sibling files with "
            "_grid, _error, and _scatter suffixes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_name, tensor = load_tensor(args)
    tensor = tensor.detach().float().cpu()
    blocks, _ = to_blocks(tensor, args.block_size)

    block_index = (
        args.block_index
        if args.block_index is not None
        else select_informative_block(
            blocks=blocks,
            search_offsets=args.search_offsets,
            max_blocks=args.max_blocks,
            seed=args.seed,
        )
    )
    if block_index < 0 or block_index >= blocks.shape[0]:
        raise ValueError(f"block index {block_index} is outside [0, {blocks.shape[0]})")

    block = blocks[block_index]
    result = compare_mxfp4_scales(block, args.search_offsets)
    output_paths = plot_scale_alignment(
        tensor_name=tensor_name,
        block_index=block_index,
        block=block,
        result=result,
        bins=args.bins,
        output=args.output,
    )

    default_mse = result["default_mse"].item()
    searched_mse = result["searched_mse"].item()
    improvement = 100.0 * (default_mse - searched_mse) / default_mse if default_mse else 0.0
    print("Saved figures:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print(f"Tensor: {tensor_name}")
    print(f"Block index: {block_index}")
    print(f"Default scale: {result['default_scale'].item():.6g}")
    print(f"Searched scale: {result['searched_scale'].item():.6g}")
    print(f"Block MSE: {default_mse:.6g} -> {searched_mse:.6g} ({improvement:.2f}% lower)")


def load_tensor(args: argparse.Namespace) -> tuple[str, torch.Tensor]:
    if args.tensor_path is not None:
        obj = torch.load(args.tensor_path, map_location="cpu")
        return tensor_from_object(obj, args.tensor_name, args.tensor_regex)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Loading Hugging Face models requires transformers.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    try:
        return tensor_from_object(model.state_dict(), args.tensor_name, args.tensor_regex)
    finally:
        del model


def tensor_from_object(
    obj: Any,
    tensor_name: str | None,
    tensor_regex: str,
) -> tuple[str, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        if tensor_name is not None:
            raise ValueError("--tensor-name is not valid when --tensor-path points to a raw tensor")
        return "tensor", obj

    if not isinstance(obj, dict):
        raise TypeError("Expected --tensor-path to contain a tensor or a state-dict-like mapping")

    if tensor_name is not None:
        tensor = obj[tensor_name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{tensor_name!r} is not a tensor")
        return tensor_name, tensor

    pattern = re.compile(tensor_regex)
    for name, value in obj.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and pattern.search(name):
            return name, value
    raise ValueError(f"No 2D tensor matched regex: {tensor_regex}")


def select_informative_block(
    *,
    blocks: torch.Tensor,
    search_offsets: list[int],
    max_blocks: int,
    seed: int,
) -> int:
    if blocks.shape[0] <= max_blocks:
        candidate_indices = torch.arange(blocks.shape[0])
    else:
        generator = torch.Generator().manual_seed(seed)
        candidate_indices = torch.randperm(blocks.shape[0], generator=generator)[:max_blocks]

    sampled_blocks = blocks[candidate_indices]
    default_scale = default_mxfp4_scale(sampled_blocks)
    default_quantized = quantize_e2m1(sampled_blocks / default_scale) * default_scale
    default_mse = torch.mean((sampled_blocks - default_quantized).square(), dim=-1)

    searched_quantized, searched_scale = search_mxfp4_scale(sampled_blocks, search_offsets)
    del searched_scale
    searched_mse = torch.mean((sampled_blocks - searched_quantized).square(), dim=-1)

    improvement = default_mse - searched_mse
    nontrivial = sampled_blocks.abs().amax(dim=-1) > 0
    improvement = torch.where(nontrivial, improvement, torch.full_like(improvement, -math.inf))
    best_sample_index = int(torch.argmax(improvement).item())
    return int(candidate_indices[best_sample_index].item())


def compare_mxfp4_scales(block: torch.Tensor, search_offsets: list[int]) -> dict[str, torch.Tensor]:
    block = block.reshape(1, -1)
    default_scale = default_mxfp4_scale(block)
    default_quantized = quantize_e2m1(block / default_scale) * default_scale
    searched_quantized, searched_scale = search_mxfp4_scale(block, search_offsets)

    return {
        "default_scale": default_scale.squeeze(),
        "searched_scale": searched_scale.squeeze(),
        "default_quantized": default_quantized.squeeze(),
        "searched_quantized": searched_quantized.squeeze(),
        "default_mse": torch.mean((block - default_quantized).square()).squeeze(),
        "searched_mse": torch.mean((block - searched_quantized).square()).squeeze(),
    }


def default_mxfp4_scale(blocks: torch.Tensor) -> torch.Tensor:
    raw_scale = raw_e2m1_block_scale(blocks, eps=1e-12)
    exponent = torch.ceil(torch.log2(raw_scale))
    return torch.pow(2.0, exponent)


def search_mxfp4_scale(
    blocks: torch.Tensor,
    search_offsets: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_scale = raw_e2m1_block_scale(blocks, eps=1e-12)
    base_exponent = torch.ceil(torch.log2(raw_scale))

    best_quantized: torch.Tensor | None = None
    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for offset in search_offsets:
        scale = torch.pow(2.0, base_exponent + offset)
        quantized = quantize_e2m1(blocks / scale) * scale
        mse = torch.mean((blocks - quantized).square(), dim=-1)
        if best_mse is None or best_quantized is None or best_scale is None:
            best_mse = mse
            best_quantized = quantized
            best_scale = scale
            continue

        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_quantized = torch.where(improved.unsqueeze(-1), quantized, best_quantized)
        best_scale = torch.where(improved.unsqueeze(-1), scale, best_scale)

    if best_quantized is None or best_scale is None:
        raise ValueError("At least one search offset is required")
    return best_quantized, best_scale


def plot_scale_alignment(
    *,
    tensor_name: str,
    block_index: int,
    block: torch.Tensor,
    result: dict[str, torch.Tensor],
    bins: int,
    output: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc

    values = block.cpu().numpy()
    default_levels = mxfp4_levels(result["default_scale"].item()).cpu().numpy()
    searched_levels = mxfp4_levels(result["searched_scale"].item()).cpu().numpy()
    default_quantized = result["default_quantized"].cpu().numpy()
    searched_quantized = result["searched_quantized"].cpu().numpy()

    output_paths = split_output_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        plot_grid_alignment(
            ax,
            values=values,
            default_levels=default_levels,
            searched_levels=searched_levels,
            default_scale=result["default_scale"].item(),
            searched_scale=result["searched_scale"].item(),
        )
        fig.savefig(output_paths["grid"], bbox_inches="tight", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        plot_error_comparison(
            ax,
            values=values,
            default_quantized=default_quantized,
            searched_quantized=searched_quantized,
            default_mse=result["default_mse"].item(),
            searched_mse=result["searched_mse"].item(),
        )
        fig.savefig(output_paths["error"], bbox_inches="tight", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        plot_original_quantized_scatter(
            ax,
            values=values,
            default_quantized=default_quantized,
            searched_quantized=searched_quantized,
        )
        fig.savefig(output_paths["scatter"], bbox_inches="tight", dpi=300)
        plt.close(fig)

    del tensor_name, block_index, bins
    return list(output_paths.values())


def plot_grid_alignment(
    ax: Any,
    *,
    values: Any,
    default_levels: Any,
    searched_levels: Any,
    default_scale: float,
    searched_scale: float,
) -> None:
    bins = min(24, max(8, len(values) // 2))
    _, _, patches = ax.hist(values, bins=bins, color="#D0D0D0", edgecolor="white", alpha=0.95)
    ymax = max((patch.get_height() for patch in patches), default=1.0)
    default_y0, default_y1 = -0.14 * ymax, -0.04 * ymax
    searched_y0, searched_y1 = -0.30 * ymax, -0.20 * ymax

    ax.vlines(default_levels, default_y0, default_y1, color="#4C78A8", linewidth=2.0, alpha=0.9)
    ax.vlines(
        searched_levels,
        searched_y0,
        searched_y1,
        color="#F58518",
        linewidth=2.0,
        alpha=0.95,
        linestyle="--",
    )
    ax.set_ylim(searched_y0 - 0.05 * ymax, ymax * 1.05)
    ax.set_xlabel("Tensor value")
    ax.set_ylabel("Count")
    ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
    default_handle = ax.plot([], [], color="#4C78A8", linewidth=2.0, label="Default")[0]
    searched_handle = ax.plot(
        [],
        [],
        color="#F58518",
        linewidth=2.0,
        linestyle="--",
        label="Searched",
    )[0]
    ax.legend(handles=[default_handle, searched_handle], frameon=False, loc="upper left")
    del default_scale, searched_scale


def plot_error_comparison(
    ax: Any,
    *,
    values: Any,
    default_quantized: Any,
    searched_quantized: Any,
    default_mse: float,
    searched_mse: float,
) -> None:
    default_error = abs(values - default_quantized)
    searched_error = abs(values - searched_quantized)
    order = default_error.argsort()[::-1]
    x_positions = range(len(values))

    ax.plot(x_positions, default_error[order], marker="x", color="#4C78A8", label="Default")
    ax.plot(x_positions, searched_error[order], marker="+", color="#F58518", label="Searched")
    ax.fill_between(
        x_positions,
        searched_error[order],
        default_error[order],
        where=default_error[order] >= searched_error[order],
        color="#F58518",
        alpha=0.16,
        interpolate=True,
    )
    ax.set_xlabel("Element rank")
    ax.set_ylabel("Absolute error")
    ax.legend(frameon=False, loc="upper right")
    del default_mse, searched_mse


def plot_original_quantized_scatter(
    ax: Any,
    *,
    values: Any,
    default_quantized: Any,
    searched_quantized: Any,
) -> None:
    lower = min(float(values.min()), float(default_quantized.min()), float(searched_quantized.min()))
    upper = max(float(values.max()), float(default_quantized.max()), float(searched_quantized.max()))
    margin = 0.06 * (upper - lower)
    lower -= margin
    upper += margin

    ax.plot([lower, upper], [lower, upper], color="#777777", linewidth=1.2, linestyle=":")
    ax.scatter(values, default_quantized, label="Default", marker="x", s=46, color="#4C78A8")
    ax.scatter(values, searched_quantized, label="Searched", marker="+", s=52, color="#F58518")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Original value")
    ax.set_ylabel("Quantized value")
    ax.legend(frameon=False, loc="upper left")


def plot_alignment_errors(
    ax: Any,
    *,
    values: Any,
    quantized: Any,
    levels: Any,
    scale: float,
    mse: float,
    color: str,
    xlim: tuple[float, float],
) -> None:
    order = values.argsort()
    sorted_values = values[order]
    sorted_quantized = quantized[order]
    y_positions = range(len(sorted_values))

    for level in levels:
        ax.axvline(level, color=color, linewidth=1.0, alpha=0.25, zorder=0)

    ax.hlines(
        y_positions,
        sorted_values,
        sorted_quantized,
        color="#A0A0A0",
        linewidth=1.1,
        alpha=0.75,
        zorder=1,
    )
    ax.scatter(sorted_values, y_positions, label="Original", s=34, color="#222222", zorder=3)
    ax.scatter(
        sorted_quantized,
        y_positions,
        label="Quantized",
        marker="x",
        s=42,
        color=color,
        zorder=4,
    )
    ax.axvline(0.0, color="#777777", linewidth=1.0, alpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Tensor value")
    ax.set_ylabel("Sorted element")
    ax.set_yticks([])
    ax.legend(frameon=False, handlelength=1.2, loc="lower right")
    ax.text(
        0.02,
        0.96,
        f"scale={scale:.3g}\nMSE={mse:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )


def value_xlim(values: Any, default_levels: Any, searched_levels: Any) -> tuple[float, float]:
    lower = min(float(values.min()), float(default_levels.min()), float(searched_levels.min()))
    upper = max(float(values.max()), float(default_levels.max()), float(searched_levels.max()))
    margin = 0.06 * (upper - lower)
    return lower - margin, upper + margin


def split_output_paths(output: Path) -> dict[str, Path]:
    suffix = output.suffix or ".pdf"
    stem = output.stem if output.suffix else output.name
    parent = output.parent
    return {
        "grid": parent / f"{stem}_grid{suffix}",
        "error": parent / f"{stem}_error{suffix}",
        "scatter": parent / f"{stem}_scatter{suffix}",
    }


def mxfp4_levels(scale: float) -> torch.Tensor:
    positive = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, E2M1_MAX])
    levels = torch.cat([-positive.flip(0), positive[1:]]) * scale
    return torch.unique(levels).sort().values


if __name__ == "__main__":
    main()
