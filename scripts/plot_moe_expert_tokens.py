#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


EXPERT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\..*\.experts\.(?P<expert>\d+)\.(?P<projection>[^.]+)$")


@dataclass(frozen=True)
class ExpertTokenStats:
    counts: list[list[int]]
    layer_indices: list[int]
    expert_indices: list[int]
    projection: str
    matched_layers: int
    matched_experts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot routed token coverage across Qwen3-MoE experts from calibration metadata."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/calib/qwen3_moe_wikitext2/metadata.json"),
        help="Calibration metadata.json containing token_counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/qwen3_moe_expert_tokens.pdf"),
        help="Output PDF path. A PNG sibling is also written unless --png-output is provided.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to the PDF path with .png suffix.",
    )
    parser.add_argument(
        "--projection",
        default="gate_up_proj",
        help="Expert projection to plot from token_counts, usually gate_up_proj.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[128, 512, 1024],
        help="Token-count thresholds summarized in the figure.",
    )
    parser.add_argument("--title", default="Qwen3-MoE Calibration Expert Coverage")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap for the heatmap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata)
    stats = expert_token_stats(metadata, projection=args.projection)
    png_output = args.png_output or args.output.with_suffix(".png")

    plot_expert_token_stats(
        stats=stats,
        metadata=metadata,
        thresholds=args.thresholds,
        title=args.title,
        cmap=args.cmap,
        pdf_output=args.output,
        png_output=png_output,
    )

    summary = summarize_counts(flatten(stats.counts), thresholds=args.thresholds)
    print(f"Saved PDF to {args.output}")
    print(f"Saved PNG to {png_output}")
    print(
        "Coverage: "
        f"layers={stats.matched_layers}, experts={stats.matched_experts}, "
        f"min={summary['min']:.0f}, p5={summary['p5']:.0f}, "
        f"median={summary['median']:.0f}, mean={summary['mean']:.1f}, max={summary['max']:.0f}"
    )
    for threshold in args.thresholds:
        below = summary[f"below_{threshold}"]
        total = summary["total"]
        print(f"Below {threshold}: {below:.0f}/{total:.0f} ({100.0 * below / total:.2f}%)")


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata JSON object: {path}")
    return metadata


def configure_matplotlib_cache() -> None:
    temp_root = Path(tempfile.gettempdir())
    mpl_cache = temp_root / "sepquant-matplotlib"
    xdg_cache = temp_root / "sepquant-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def expert_token_stats(metadata: dict[str, Any], *, projection: str) -> ExpertTokenStats:
    token_counts = metadata.get("token_counts")
    if not isinstance(token_counts, dict):
        raise ValueError("metadata does not contain a token_counts object")

    parsed: dict[tuple[int, int], int] = {}
    layer_set: set[int] = set()
    expert_set: set[int] = set()
    for name, count in token_counts.items():
        if not isinstance(name, str):
            continue
        match = EXPERT_LAYER_RE.search(name)
        if match is None or match.group("projection") != projection:
            continue

        layer_idx = int(match.group("layer"))
        expert_idx = int(match.group("expert"))
        parsed[(layer_idx, expert_idx)] = int(count)
        layer_set.add(layer_idx)
        expert_set.add(expert_idx)

    if not parsed:
        raise ValueError(f"No expert token counts found for projection={projection!r}")

    layer_indices = list(range(max(layer_set) + 1))
    expert_indices = list(range(max(expert_set) + 1))
    counts = [
        [parsed.get((layer_idx, expert_idx), 0) for expert_idx in expert_indices]
        for layer_idx in layer_indices
    ]
    return ExpertTokenStats(
        counts=counts,
        layer_indices=layer_indices,
        expert_indices=expert_indices,
        projection=projection,
        matched_layers=len(layer_set),
        matched_experts=len(expert_set),
    )


def plot_expert_token_stats(
    *,
    stats: ExpertTokenStats,
    metadata: dict[str, Any],
    thresholds: list[int],
    title: str,
    cmap: str,
    pdf_output: Path,
    png_output: Path,
) -> None:
    configure_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc

    values = stats.counts
    flattened = flatten(values)
    summary = summarize_counts(flattened, thresholds=thresholds)
    layer_min, layer_p10, layer_median = layer_summaries(values)
    expert_mean, expert_min = expert_summaries(values)
    positive_values = [value for value in flattened if value > 0]
    vmax = max(flattened) if flattened else 1
    vmin = max(1, min(positive_values) if positive_values else 1)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1))

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig = plt.figure(figsize=(7.0, 4.8), constrained_layout=True)
        grid = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=[5.4, 0.18, 1.55],
            height_ratios=[4.0, 1.25],
        )
        ax_heatmap = fig.add_subplot(grid[0, 0])
        ax_colorbar = fig.add_subplot(grid[0, 1])
        ax_layer = fig.add_subplot(grid[0, 2])
        ax_expert = fig.add_subplot(grid[1, 0], sharex=ax_heatmap)
        ax_text = fig.add_subplot(grid[1, 2])

        plot_values = [[max(value, 1) for value in row] for row in values]
        image = ax_heatmap.imshow(plot_values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        colorbar = fig.colorbar(image, cax=ax_colorbar)
        colorbar.set_label("Tokens (log)")

        ax_heatmap.set_title(title, pad=8)
        ax_heatmap.set_ylabel("Layer")
        ax_heatmap.set_xlabel("Expert index")
        ax_heatmap.set_xticks(tick_positions(len(stats.expert_indices), step=16))
        ax_heatmap.set_yticks(tick_positions(len(stats.layer_indices), step=8))
        ax_heatmap.tick_params(axis="x", labelbottom=False)
        ax_heatmap.grid(False)

        y_positions = list(range(len(stats.layer_indices)))
        ax_layer.plot(layer_median, y_positions, color="#4C78A8", label="median")
        ax_layer.plot(layer_p10, y_positions, color="#F58518", linestyle="--", label="p10")
        ax_layer.plot(layer_min, y_positions, color="#777777", linestyle=":", label="min")
        ax_layer.invert_yaxis()
        ax_layer.set_xlabel("Tokens")
        ax_layer.set_title("Per-layer")
        ax_layer.grid(axis="x", color="#D8D8D8", linewidth=0.7, alpha=0.8)
        ax_layer.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

        x_positions = list(range(len(stats.expert_indices)))
        ax_expert.plot(x_positions, expert_mean, color="#4C78A8", label="mean")
        ax_expert.plot(x_positions, expert_min, color="#777777", linestyle=":", label="min")
        ax_expert.set_ylabel("Tokens")
        ax_expert.set_xlabel("Expert index")
        ax_expert.set_xticks(tick_positions(len(stats.expert_indices), step=16))
        ax_expert.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
        ax_expert.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28))

        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            summary_text(
                metadata=metadata,
                stats=stats,
                summary=summary,
                thresholds=thresholds,
            ),
            ha="left",
            va="top",
            linespacing=1.35,
            transform=ax_text.transAxes,
        )

        for axis in (ax_heatmap, ax_layer, ax_expert):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight", dpi=300)
        fig.savefig(png_output, bbox_inches="tight", dpi=300)
        plt.close(fig)


def layer_summaries(values: list[list[int]]) -> tuple[list[float], list[float], list[float]]:
    layer_min = []
    layer_p10 = []
    layer_median = []
    for row in values:
        layer_min.append(float(min(row)))
        layer_p10.append(percentile(row, 10.0))
        layer_median.append(float(median(row)))
    return layer_min, layer_p10, layer_median


def expert_summaries(values: list[list[int]]) -> tuple[list[float], list[float]]:
    columns = transpose(values)
    return [float(mean(column)) for column in columns], [float(min(column)) for column in columns]


def summarize_counts(values: list[int], *, thresholds: list[int]) -> dict[str, float]:
    if not values:
        raise ValueError("No token counts to summarize")
    summary: dict[str, float] = {
        "total": float(len(values)),
        "min": float(min(values)),
        "p1": percentile(values, 1.0),
        "p5": percentile(values, 5.0),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }
    for threshold in thresholds:
        summary[f"below_{threshold}"] = float(sum(value < threshold for value in values))
    return summary


def summary_text(
    *,
    metadata: dict[str, Any],
    stats: ExpertTokenStats,
    summary: dict[str, float],
    thresholds: list[int],
) -> str:
    nsamples = metadata.get("nsamples", "?")
    batch_size = metadata.get("batch_size", "?")
    sequence_length = metadata.get("sequence_length", "?")
    lines = [
        f"Projection: {stats.projection}",
        f"Layers x experts: {len(stats.layer_indices)} x {len(stats.expert_indices)}",
        f"Samples: {nsamples}, batch: {batch_size}, seq: {sequence_length}",
        "",
        f"min / p5 / median: {summary['min']:.0f} / {summary['p5']:.0f} / {summary['median']:.0f}",
        f"mean / max: {summary['mean']:.1f} / {summary['max']:.0f}",
    ]
    total = summary["total"]
    for threshold in thresholds:
        below = summary[f"below_{threshold}"]
        lines.append(f"< {threshold}: {below:.0f} ({100.0 * below / total:.1f}%)")
    return "\n".join(lines)


def tick_positions(length: int, *, step: int) -> list[int]:
    if length <= 0:
        return []
    ticks = list(range(0, length, step))
    last = length - 1
    if ticks[-1] != last:
        ticks.append(last)
    return ticks


def flatten(values: list[list[int]]) -> list[int]:
    return [value for row in values for value in row]


def transpose(values: list[list[int]]) -> list[list[int]]:
    if not values:
        return []
    return [[row[column_idx] for row in values] for column_idx in range(len(values[0]))]


def percentile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


if __name__ == "__main__":
    main()
