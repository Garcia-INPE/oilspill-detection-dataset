#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import resolve_benchmark_dir


NUMERIC_FIELDS = [
    "primary_metric_value",
    "mean_iou_fg",
    "iou_slick",
    "iou_oil_spill",
    "iou_seepage",
    "mean_iou_all",
    "pixel_accuracy",
    "loss",
]

TEXT_FIELDS = [
    "benchmark_id",
    "timestamp_utc",
    "class_map_version",
    "model_name",
    "primary_metric_name",
    "device",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark rounds from BENCHMARK_TABLE.csv")
    parser.add_argument("--benchmark-a", required=True, help="First benchmark id")
    parser.add_argument("--benchmark-b", required=True, help="Second benchmark id")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the comparison report as Markdown",
    )
    return parser.parse_args()

def load_rows(table_path: Path) -> dict[str, dict[str, str]]:
    if not table_path.exists():
        raise FileNotFoundError(f"Benchmark table not found: {table_path}")

    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {row["benchmark_id"]: row for row in reader}

    if not rows:
        raise ValueError(f"Benchmark table is empty: {table_path}")

    return rows


def get_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_value(value: str | None) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def format_delta(delta: float | None) -> str:
    if delta is None or not math.isfinite(delta):
        return "-"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6f}"


def format_percent(delta: float | None, base: float | None) -> str:
    if delta is None or base is None or base == 0 or not math.isfinite(delta) or not math.isfinite(base):
        return "-"
    pct = (delta / base) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def compare_rows(row_a: dict[str, str], row_b: dict[str, str]) -> list[tuple[str, str, str, str, str]]:
    report_rows: list[tuple[str, str, str, str, str]] = []

    for field in TEXT_FIELDS:
        value_a = row_a.get(field, "")
        value_b = row_b.get(field, "")
        if value_a != value_b and field != "benchmark_id":
            report_rows.append((field, format_value(value_a), format_value(value_b), "", ""))

    for field in NUMERIC_FIELDS:
        value_a = get_float(row_a.get(field))
        value_b = get_float(row_b.get(field))
        if value_a is None and value_b is None:
            continue
        if value_a is None or value_b is None:
            report_rows.append((field, format_value(row_a.get(field)), format_value(row_b.get(field)), "-", "-"))
            continue

        delta = value_b - value_a
        report_rows.append(
            (
                field,
                f"{value_a:.6f}",
                f"{value_b:.6f}",
                format_delta(delta),
                format_percent(delta, value_a),
            )
        )

    return report_rows


def build_markdown_report(row_a: dict[str, str], row_b: dict[str, str], rows: list[tuple[str, str, str, str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Comparison")
    lines.append("")
    lines.append(f"- A: {row_a.get('benchmark_id', '-')}")
    lines.append(f"- B: {row_b.get('benchmark_id', '-')}")
    lines.append("")
    lines.append("| Field | A | B | Delta (B-A) | Delta % |")
    lines.append("| --- | --- | --- | --- | --- |")
    for field, value_a, value_b, delta, delta_pct in rows:
        lines.append(f"| {field} | {value_a} | {value_b} | {delta} | {delta_pct} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    benchmark_dir = resolve_benchmark_dir()
    benchmark_table = benchmark_dir / "BENCHMARK_TABLE.csv"

    rows = load_rows(benchmark_table)

    benchmark_a = args.benchmark_a
    benchmark_b = args.benchmark_b

    if benchmark_a not in rows:
        raise KeyError(f"Benchmark not found in table: {benchmark_a}")
    if benchmark_b not in rows:
        raise KeyError(f"Benchmark not found in table: {benchmark_b}")

    row_a = rows[benchmark_a]
    row_b = rows[benchmark_b]

    comparison_rows = compare_rows(row_a, row_b)
    report = build_markdown_report(row_a, row_b, comparison_rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote comparison report to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()