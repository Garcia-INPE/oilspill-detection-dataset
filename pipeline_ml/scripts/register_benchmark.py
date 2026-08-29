#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import resolve_results_dir, resolve_benchmark_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate visual evaluation and register a benchmark run with <benchmark_id> snapshots"
    )
    p.add_argument("--run-name", type=str, default="test_reference")
    p.add_argument("--benchmark-id", type=str, required=True)
    p.add_argument("--model-name", type=str, default="TinyUNet")
    p.add_argument("--notes", type=str, default="Initial baseline benchmark")
    p.add_argument("--samples-per-page", type=int, default=2)
    p.add_argument(
        "--generate-visual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PDF/CSV visual evaluation before registering benchmark",
    )
    return p.parse_args()


def read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_existing_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["benchmark_id"] for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_id",
        "timestamp_utc",
        "class_map_version",
        "model_name",
        "primary_metric_name",
        "primary_metric_value",
        "mean_iou_fg",
        "iou_slick",
        "iou_oil_spill",
        "iou_seepage",
        "mean_iou_all",
        "pixel_accuracy",
        "loss",
        "device",
        "notes",
    ]

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    # If schema changed, rewrite table preserving old rows and adding missing columns.
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_fields = reader.fieldnames or []
        existing_rows = list(reader)

    if current_fields != fieldnames:
        normalized_rows = []
        for old in existing_rows:
            normalized = {k: old.get(k, "") for k in fieldnames}
            normalized_rows.append(normalized)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(normalized_rows)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def move_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def resolve_checkpoint_path(project_root: Path, checkpoint_value: str) -> Path:
    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return project_root / checkpoint_path


def main() -> None:
    args = parse_args()

    results_dir = resolve_results_dir()
    benchmark_dir = resolve_benchmark_dir()

    project_root = Path(__file__).resolve().parents[1]
    class_map_version = "v1"
    run_dir = results_dir / args.run_name
    table_path = benchmark_dir / "BENCHMARK_TABLE.csv"
    existing_ids = load_existing_ids(table_path)
    row_id = args.benchmark_id
    if row_id in existing_ids:
        raise ValueError(f"benchmark_id already exists in table: {row_id}")

    metrics_path = run_dir / "metrics_test.json"
    test_samples_path = run_dir / "test_samples.csv"
    test_metrics_snapshot = run_dir / f"{args.benchmark_id}_metrics_test.json"
    test_samples_snapshot = run_dir / f"{args.benchmark_id}_test_samples.csv"
    test_pdf_snapshot = run_dir / f"{args.benchmark_id}_test_evaluation.pdf"
    test_summary_snapshot = run_dir / f"{args.benchmark_id}_test_evaluation_summary.csv"

    if args.generate_visual:
        script_path = Path(__file__).resolve().parent / "visualize_test_evaluation.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--test-run-name",
            args.run_name,
            "--samples-per-page",
            str(args.samples_per_page),
            "--output-pdf",
            str(test_pdf_snapshot),
            "--output-csv",
            str(test_summary_snapshot),
        ]
        subprocess.run(cmd, check=True)

    if not metrics_path.exists() and not test_metrics_snapshot.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path} (or tagged version: {test_metrics_snapshot})"
        )

    if metrics_path.exists():
        move_if_exists(metrics_path, test_metrics_snapshot)
    metrics = read_metrics(test_metrics_snapshot)

    val_run_dir = results_dir / "val_reference"
    train_run_dir = results_dir / "train_reference"
    val_metrics_path = val_run_dir / "metrics_val.json"
    val_metrics_snapshot = val_run_dir / f"{args.benchmark_id}_metrics_val.json"
    train_history_path = train_run_dir / "history.json"
    train_history_snapshot = train_run_dir / f"{args.benchmark_id}_history.json"

    checkpoint_value = str(metrics.get("checkpoint", "")).strip()
    train_checkpoint_path = (
        resolve_checkpoint_path(project_root, checkpoint_value) if checkpoint_value else train_run_dir / "checkpoints" / "best.pt"
    )
    train_checkpoint_snapshot = train_checkpoint_path.parent / f"{args.benchmark_id}_{train_checkpoint_path.name}"
    iou = metrics.get("iou_per_class", [])

    benchmark_dir_obj = benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Move predictions into a benchmark-tagged subdirectory (no duplication).
    predictions_src = run_dir / "predictions"
    predictions_dst = run_dir / "predictions" / args.benchmark_id
    if predictions_src.exists():
        if predictions_dst.exists():
            raise FileExistsError(f"Benchmark predictions directory already exists: {predictions_dst}")
        predictions_dst.mkdir(parents=True, exist_ok=False)
        pred_files = sorted(predictions_src.glob("*_pred_mask.png"))
        if not pred_files:
            raise FileNotFoundError(
                f"No root prediction masks found in {predictions_src}. Run test first to generate predictions."
            )
        for pred_file in pred_files:
            shutil.move(str(pred_file), str(predictions_dst / pred_file.name))

    # Keep test artifacts as <benchmark_id>_* only (no duplicate untagged copies after registration).
    created = []
    if move_if_exists(val_metrics_path, val_metrics_snapshot):
        created.append(str(val_metrics_snapshot))
    if move_if_exists(train_history_path, train_history_snapshot):
        created.append(str(train_history_snapshot))
    if move_if_exists(train_checkpoint_path, train_checkpoint_snapshot):
        created.append(str(train_checkpoint_snapshot))
    if move_if_exists(test_samples_path, test_samples_snapshot):
        created.append(str(test_samples_snapshot))
    if test_pdf_snapshot.exists():
        created.append(str(test_pdf_snapshot))
    if test_summary_snapshot.exists():
        created.append(str(test_summary_snapshot))
    if test_metrics_snapshot.exists():
        created.append(str(test_metrics_snapshot))

    iou_slick = float(iou[0]) if len(iou) > 0 else float("nan")
    iou_oil = ""
    iou_seepage = ""
    primary_metric_name = "iou_slick"
    primary_metric_value = iou_slick

    row = {
        "benchmark_id": row_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "class_map_version": class_map_version,
        "model_name": args.model_name,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": str(primary_metric_value),
        "mean_iou_fg": str(metrics.get("mean_iou", "")),
        "iou_slick": str(iou_slick),
        "iou_oil_spill": str(iou_oil),
        "iou_seepage": str(iou_seepage),
        "mean_iou_all": str(metrics.get("mean_iou_all", "")),
        "pixel_accuracy": str(metrics.get("pixel_accuracy", "")),
        "loss": str(metrics.get("loss", "")),
        "device": str(metrics.get("device", "")),
        "notes": args.notes,
    }
    append_row(table_path, row)

    print(f"Registered benchmark in table: {table_path}")
    print(f"Benchmark id: {row_id}")
    print("Snapshot artifacts:")
    for p in created:
        print(f"- {p}")


if __name__ == "__main__":
    main()
