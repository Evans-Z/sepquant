#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPLResult:
    format: str
    method: str
    ppl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot perplexity comparison across FP4 formats and quantization methods."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/results/ppl.md"),
        help="Markdown table containing Format, Methods, and PPL columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ppl_comparison.pdf"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["MXFP4", "NVFP4", "HiF4"],
        help="Format order on the x-axis.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RTN", "ScaleSearch", "SoScale"],
        help="Methods to plot, in legend order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = read_markdown_table(args.input)
    bf16_ppl = get_bf16_ppl(results)

    values = {
        method: [
            get_ppl(results, quant_format=quant_format, method=method)
            for quant_format in args.formats
        ]
        for method in args.methods
    }

    plot_ppl_comparison(
        formats=args.formats,
        values=values,
        bf16_ppl=bf16_ppl,
        output=args.output,
    )

    print(f"Saved figure to {args.output}")
    if bf16_ppl is not None:
        print(f"BF16 PPL: {bf16_ppl:.4g}")
    for method, method_values in values.items():
        formatted = ", ".join(
            f"{quant_format}={ppl:.4g}" for quant_format, ppl in zip(args.formats, method_values)
        )
        print(f"{method}: {formatted}")


def read_markdown_table(path: Path) -> list[PPLResult]:
    rows: list[PPLResult] = []
    with path.open("r", encoding="utf-8") as handle:
        table_rows = [
            parse_markdown_row(line)
            for line in handle
            if line.strip() and not is_markdown_separator(line)
        ]
        if not table_rows:
            raise ValueError(f"No rows found in {path}")

        header = table_rows[0]
        normalized_header = [cell.lower() for cell in header]
        try:
            format_idx = normalized_header.index("format")
            method_idx = normalized_header.index("methods")
            ppl_idx = normalized_header.index("ppl")
        except ValueError as exc:
            raise ValueError("Expected columns: Format, Methods, PPL") from exc

        for row in table_rows[1:]:
            if len(row) <= max(format_idx, method_idx, ppl_idx):
                continue
            rows.append(
                PPLResult(
                    format=row[format_idx].strip(),
                    method=row[method_idx].strip(),
                    ppl=float(row[ppl_idx].strip()),
                )
            )
    return rows


def parse_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_markdown_separator(line: str) -> bool:
    cells = parse_markdown_row(line)
    return bool(cells) and all(set(cell) <= {"-", ":"} for cell in cells)


def get_bf16_ppl(results: list[PPLResult]) -> float | None:
    for result in results:
        if result.format.upper() == "BF16":
            return result.ppl
    return None


def get_ppl(results: list[PPLResult], *, quant_format: str, method: str) -> float:
    for result in results:
        if result.format == quant_format and result.method == method:
            return result.ppl
    raise ValueError(f"Missing PPL for format={quant_format!r}, method={method!r}")


def plot_ppl_comparison(
    *,
    formats: list[str],
    values: dict[str, list[float]],
    bf16_ppl: float | None,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc

    x_positions = list(range(len(formats)))
    styles = {
        "RTN": {"color": "#4C78A8", "marker": "x", "linestyle": "-"},
        "ScaleSearch": {"color": "#F58518", "marker": "+", "linestyle": "--"},
        "SoScale": {"color": "#54A24B", "marker": "o", "linestyle": "-"},
    }

    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        for method, method_values in values.items():
            style = styles.get(method, {"marker": "o", "linestyle": "-"})
            ax.plot(
                x_positions,
                method_values,
                label=method,
                markersize=8,
                markeredgewidth=2.0,
                **style,
            )

        if bf16_ppl is not None:
            ax.axhline(
                bf16_ppl,
                color="#777777",
                linewidth=1.8,
                linestyle=":",
                label="BF16",
            )

        ax.set_xticks(x_positions, formats)
        ax.set_ylabel("Perplexity")
        ax.set_xlabel("Format")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
        ax.legend(frameon=False, loc="upper right")
        ax.margins(x=0.08)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
