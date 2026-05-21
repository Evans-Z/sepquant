from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML summary from SepQuant experiment runs.")
    parser.add_argument("experiment_dir", type=Path, help="Directory containing experiment run folders.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.experiment_dir / "report.html")
    runs = load_runs(args.experiment_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(runs=runs, title="SepQuant Experiments"), encoding="utf-8")
    print(f"Saved experiment report to {output_path}")


def load_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for metrics_path in sorted(experiment_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        runs.append(
            {
                "run": run_dir.name,
                "path": str(run_dir),
                "metrics": _read_json(metrics_path),
                "config": _read_json(run_dir / "config.json"),
                "metadata": _read_json(run_dir / "metadata.json"),
                "artifacts": _read_json(run_dir / "artifacts.json"),
            }
        )
    return runs


def render_html(*, runs: list[dict[str, Any]], title: str) -> str:
    rows = "\n".join(_render_row(run) for run in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #5d6678; margin-top: 0; }}
    input {{ box-sizing: border-box; width: 100%; padding: 10px 12px; margin: 18px 0; border: 1px solid #ccd3df; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f7f9fc; cursor: pointer; user-select: none; }}
    tr:hover {{ background: #fbfcff; }}
    code {{ background: #f1f4f8; padding: 2px 5px; border-radius: 4px; }}
    details {{ max-width: 520px; }}
    pre {{ white-space: pre-wrap; background: #f7f9fc; border: 1px solid #e1e6ef; border-radius: 8px; padding: 10px; max-height: 360px; overflow: auto; }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .empty {{ color: #8a94a6; }}
  </style>
</head>
<body>
  <h1>{_escape(title)}</h1>
  <p class="subtitle">{len(runs)} run(s). Generated from local JSON experiment records.</p>
  <input id="filter" placeholder="Filter by model, weight source, dataset, method, metric, path..." oninput="filterRows()">
  <table id="runs">
    <thead>
      <tr>
        <th onclick="sortTable(0)">Run</th>
        <th onclick="sortTable(1)">Type</th>
        <th onclick="sortTable(2)">Timestamp</th>
        <th onclick="sortTable(3)">Model</th>
        <th onclick="sortTable(4)">Weight Source</th>
        <th onclick="sortTable(5)">Mode</th>
        <th onclick="sortTable(6)">Data</th>
        <th onclick="sortTable(7)">Formats</th>
        <th onclick="sortTable(8)">PPL</th>
        <th onclick="sortTable(9)">Loss</th>
        <th onclick="sortTable(10)">Tokens</th>
        <th onclick="sortTable(11)">Task Metrics</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <script>
    function filterRows() {{
      const query = document.getElementById("filter").value.toLowerCase();
      for (const row of document.querySelectorAll("#runs tbody tr")) {{
        row.style.display = row.innerText.toLowerCase().includes(query) ? "" : "none";
      }}
    }}
    function sortTable(column) {{
      const table = document.getElementById("runs");
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      const ascending = table.dataset.sortColumn != column || table.dataset.sortDirection == "desc";
      rows.sort((a, b) => compareCells(a.cells[column].innerText, b.cells[column].innerText, ascending));
      rows.forEach(row => body.appendChild(row));
      table.dataset.sortColumn = column;
      table.dataset.sortDirection = ascending ? "asc" : "desc";
    }}
    function compareCells(left, right, ascending) {{
      const leftNumber = Number(left);
      const rightNumber = Number(right);
      const result = Number.isFinite(leftNumber) && Number.isFinite(rightNumber)
        ? leftNumber - rightNumber
        : left.localeCompare(right);
      return ascending ? result : -result;
    }}
  </script>
</body>
</html>
"""


def _render_row(run: dict[str, Any]) -> str:
    config = run["config"]
    metrics = run["metrics"]
    metadata = run["metadata"]
    details = {
        "path": run["path"],
        "config": config,
        "metrics": metrics,
        "metadata": metadata,
        "artifacts": run["artifacts"],
    }
    return "      <tr>\n" + "\n".join(
        [
            _td(run["run"]),
            _td(metadata.get("run_type") or metrics.get("task")),
            _td(metadata.get("timestamp")),
            _td(config.get("model")),
            _td(_weight_source_label(config, metrics, run["artifacts"])),
            _td(_quantization_mode(config, metrics, run["artifacts"])),
            _td(_data_label(config, metrics)),
            _td(_format_label(config)),
            _td(metrics.get("ppl"), class_name="num"),
            _td(metrics.get("loss"), class_name="num"),
            _td(metrics.get("tokens"), class_name="num"),
            _td(_task_metrics(metrics)),
            _td(f"<details><summary>JSON</summary><pre>{_escape(_json(details))}</pre></details>", raw=True),
        ]
    ) + "\n      </tr>"


def _weight_source_label(config: dict[str, Any], metrics: dict[str, Any], artifacts: dict[str, Any]) -> str:
    return str(
        metrics.get("weight_source")
        or config.get("pre_quant_model")
        or artifacts.get("pre_quant_model")
        or config.get("model")
        or ""
    )


def _quantization_mode(config: dict[str, Any], metrics: dict[str, Any], artifacts: dict[str, Any]) -> str:
    mode = metrics.get("quantization_mode")
    if mode:
        return str(mode)
    if config.get("pre_quant_model") or artifacts.get("pre_quant_model"):
        return "pre_quantized"
    if config.get("quantization_plan"):
        return "runtime_plan"
    weight = config.get("weight_format")
    if weight and weight != "none":
        return "runtime"
    return "none"


def _data_label(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    if "tasks" in metrics:
        return ", ".join(str(task) for task in metrics["tasks"])
    dataset = metrics.get("dataset") or config.get("dataset")
    dataset_config = metrics.get("dataset_config") or config.get("dataset_config")
    split = metrics.get("split") or config.get("split")
    return " / ".join(str(part) for part in [dataset, dataset_config, split] if part)


def _format_label(config: dict[str, Any]) -> str:
    weight = config.get("weight_format", "none")
    activation = config.get("activation_format", "none")
    plan = config.get("quantization_plan")
    label = f"w={weight}, a={activation}"
    if plan:
        label += f"\nplan={Path(str(plan)).name}"
    return label


def _task_metrics(metrics: dict[str, Any]) -> str:
    results = metrics.get("results")
    if not isinstance(results, dict):
        return ""

    parts = []
    for task, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        numeric = {
            key: value
            for key, value in task_metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        if numeric:
            rendered = ", ".join(f"{key}={value:.4g}" for key, value in sorted(numeric.items()))
            parts.append(f"{task}: {rendered}")
    return "\n".join(parts)


def _td(value: Any, *, class_name: str | None = None, raw: bool = False) -> str:
    class_attr = f' class="{class_name}"' if class_name else ""
    if value is None or value == "":
        return f"<td{class_attr}><span class=\"empty\">-</span></td>"
    text = str(value) if raw else _escape(value)
    return f"<td{class_attr}>{text}</td>"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"value": data}


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("\n", "<br>")


if __name__ == "__main__":
    main()
