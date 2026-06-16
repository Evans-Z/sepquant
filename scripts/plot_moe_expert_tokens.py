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
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap for the heatmap.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[3.6, 2.0],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Default is 3:2 and fits roughly 0.5 text width in ICLR style.",
    )
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
        cmap=args.cmap,
        figsize=tuple(args.figsize),
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
    cmap: str,
    figsize: tuple[float, float],
    pdf_output: Path,
    png_output: Path,
) -> None:
    configure_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc

    values = stats.counts
    flattened = flatten(values)
    del metadata, thresholds
    layer_median = layer_summaries(values)
    expert_median = expert_summaries(values)
    positive_values = [value for value in flattened if value > 0]
    vmax = max(flattened) if flattened else 1
    vmin = max(1, min(positive_values) if positive_values else 1)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1))

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 12,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[3.6, 1.0],
            height_ratios=[2.25, 0.95],
        )
        ax_heatmap = fig.add_subplot(grid[0, 0])
        ax_layer = fig.add_subplot(grid[0, 1])
        ax_expert = fig.add_subplot(grid[1, 0], sharex=ax_heatmap)
        ax_legend = fig.add_subplot(grid[1, 1])

        plot_values = [[max(value, 1) for value in row] for row in values]
        image = ax_heatmap.imshow(plot_values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.035, pad=0.02)
        colorbar.set_label("Tokens")

        ax_heatmap.set_ylabel("Layer")
        ax_heatmap.set_xlabel("Expert")
        ax_heatmap.set_xticks(tick_positions(len(stats.expert_indices), step=32))
        ax_heatmap.set_yticks(tick_positions(len(stats.layer_indices), step=8))
        ax_heatmap.tick_params(axis="x", labelbottom=False)
        ax_heatmap.grid(False)

        y_positions = list(range(len(stats.layer_indices)))
        ax_layer.plot(layer_median, y_positions, color="#4C78A8", label="median")
        ax_layer.invert_yaxis()
        ax_layer.set_xlabel("Tokens")
        ax_layer.set_xlim(left=0)
        ax_layer.set_xticks([0, nice_upper_tick(layer_median)])
        ax_layer.tick_params(axis="x", labelsize=7)
        ax_layer.grid(axis="x", color="#D8D8D8", linewidth=0.7, alpha=0.8)

        x_positions = list(range(len(stats.expert_indices)))
        ax_expert.plot(x_positions, expert_median, color="#4C78A8", label="median")
        ax_expert.set_ylabel("Tokens")
        ax_expert.set_xlabel("Expert")
        ax_expert.set_xticks(tick_positions(len(stats.expert_indices), step=32))
        ax_expert.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_expert.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)

        ax_legend.axis("off")
        handles, labels = ax_expert.get_legend_handles_labels()
        ax_legend.legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            handlelength=2.2,
        )

        for axis in (ax_heatmap, ax_layer, ax_expert):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight", dpi=300)
        fig.savefig(png_output, bbox_inches="tight", dpi=300)
        plt.close(fig)


def layer_summaries(values: list[list[int]]) -> list[float]:
    layer_median = []
    for row in values:
        layer_median.append(float(median(row)))
    return layer_median


def expert_summaries(values: list[list[int]]) -> list[float]:
    columns = transpose(values)
    return [float(median(column)) for column in columns]


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


def tick_positions(length: int, *, step: int) -> list[int]:
    if length <= 0:
        return []
    ticks = list(range(0, length, step))
    last = length - 1
    if ticks[-1] != last:
        ticks.append(last)
    return ticks


def nice_upper_tick(values: list[float]) -> float:
    upper = max(values) if values else 1.0
    if upper <= 0:
        return 1.0
    exponent = math.floor(math.log10(upper))
    base = 10**exponent
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier * base
        if candidate >= upper:
            return float(candidate)
    return float(10 * base)


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
