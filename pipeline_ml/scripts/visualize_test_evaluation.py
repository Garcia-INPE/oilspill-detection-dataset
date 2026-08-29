#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import resolve_dataset_root, resolve_results_dir

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate side-by-side GT vs prediction visual evaluation for all test samples"
    )
    p.add_argument("--test-run-name", type=str, default="test_reference")
    p.add_argument(
        "--benchmark-id",
        type=str,
        default=None,
        help="Read predictions from scripts/results/<run-name>/predictions/<benchmark-id>/",
    )
    p.add_argument("--samples-per-page", type=int, default=2)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def read_test_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep first occurrence only in case of duplicate sample_id entries.
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        unique_rows.append(row)
    return unique_rows


def read_class_map() -> dict[int, str]:
    return {0: "SEA", 1: "SLICK"}


def classes_from_mask(mask_path: Path) -> list[int]:
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return sorted(int(v) for v in np.unique(arr))


def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    return (mask_np > 0).astype(np.uint8)


def class_ids_to_text(class_ids: list[int], class_map: dict[int, str]) -> str:
    # SEA/background (class 0) is omitted from reporting to focus on target classes.
    class_ids = [cid for cid in class_ids if cid != 0]
    if not class_ids:
        return "none"
    parts = []
    for cid in class_ids:
        cname = class_map.get(cid, "UNKNOWN")
        parts.append(f"{cid}:{cname}")
    return " | ".join(parts)


def pred_path_for(sample_id: str, preds_dir: Path) -> Path:
    return preds_dir / f"{sample_id}_pred_mask.png"


def split_rows_by_prediction(
    rows: list[dict[str, str]],
    preds_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    available: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        if pred_path_for(row["sample_id"], preds_dir).exists():
            available.append(row)
        else:
            missing.append(row)
    return available, missing


def write_summary_csv(rows: list[dict[str, str]], class_map: dict[int, str], preds_dir: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "gt_class_ids",
                "gt_classes",
                "pred_class_ids",
                "pred_classes",
            ],
        )
        writer.writeheader()

        for row in rows:
            sid = row["sample_id"]
            gt_abs = Path(row["mask_path"])
            pred_abs = pred_path_for(sid, preds_dir)

            gt_arr = remap_mask(np.asarray(Image.open(gt_abs).convert("L"), dtype=np.uint8))
            pred_arr = remap_mask(np.asarray(Image.open(pred_abs).convert("L"), dtype=np.uint8))
            gt_ids = sorted(int(v) for v in np.unique(gt_arr))
            pred_ids = sorted(int(v) for v in np.unique(pred_arr))
            writer.writerow(
                {
                    "sample_id": sid,
                    "gt_class_ids": ",".join(str(v) for v in gt_ids),
                    "gt_classes": class_ids_to_text(gt_ids, class_map),
                    "pred_class_ids": ",".join(str(v) for v in pred_ids),
                    "pred_classes": class_ids_to_text(pred_ids, class_map),
                }
            )


def build_pdf(
    rows: list[dict[str, str]],
    class_map: dict[int, str],
    preds_dir: Path,
    out_pdf: Path,
    samples_per_page: int,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#000000", "#00FFFF", "#FF0000"])
    max_class_id = max(class_map.keys()) if class_map else 1
    total = len(rows)
    pages = (total + samples_per_page - 1) // samples_per_page

    with PdfPages(out_pdf) as pdf:
        for page_idx in range(pages):
            page_rows = rows[page_idx * samples_per_page:(page_idx + 1) * samples_per_page]
            fig, axes = plt.subplots(
                nrows=samples_per_page,
                ncols=3,
                figsize=(16, 5 * samples_per_page),
                squeeze=False,
            )
            fig.suptitle(f"Test Set Visual Evaluation (page {page_idx + 1}/{pages})", fontsize=14, y=0.995)

            for i in range(samples_per_page):
                for j in range(3):
                    axes[i, j].axis("off")

            for row_i, row in enumerate(page_rows):
                sid = row["sample_id"]
                img = np.asarray(Image.open(Path(row["image_path"])).convert("RGB"), dtype=np.uint8)
                gt = remap_mask(np.asarray(Image.open(Path(row["mask_path"])).convert("L"), dtype=np.uint8))
                pred = remap_mask(np.asarray(Image.open(pred_path_for(sid, preds_dir)).convert("L"), dtype=np.uint8))

                gt_ids = sorted(int(v) for v in np.unique(gt))
                pred_ids = sorted(int(v) for v in np.unique(pred))

                axes[row_i, 0].imshow(img)
                axes[row_i, 0].set_title(f"{sid} - RGB", fontsize=10)
                axes[row_i, 0].axis("off")

                axes[row_i, 1].imshow(gt, cmap=cmap, vmin=0, vmax=max_class_id, interpolation="nearest")
                axes[row_i, 1].set_title(
                    "GT classes: " + class_ids_to_text(gt_ids, class_map),
                    fontsize=10,
                )
                axes[row_i, 1].axis("off")

                axes[row_i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=max_class_id, interpolation="nearest")
                axes[row_i, 2].set_title(
                    "Pred classes: " + class_ids_to_text(pred_ids, class_map),
                    fontsize=10,
                )
                axes[row_i, 2].axis("off")

            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.samples_per_page < 1 or args.samples_per_page > 4:
        raise ValueError("--samples-per-page must be between 1 and 4")

    dataset_root = resolve_dataset_root()
    results_dir = resolve_results_dir()

    split_csv = dataset_root / "splits" / "test.csv"
    results_mode_dir = results_dir / args.test_run_name
    preds_dir_base = results_mode_dir / "predictions"
    preds_subdir = args.benchmark_id if args.benchmark_id else None
    preds_dir = preds_dir_base / preds_subdir if preds_subdir else preds_dir_base

    rows_raw = read_test_rows(split_csv)
    rows: list[dict[str, str]] = []
    for row in rows_raw:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "image_path": str(dataset_root / row["image_path"]),
                "mask_path": str(dataset_root / row["mask_path"]),
            }
        )

    rows_available, rows_missing = split_rows_by_prediction(rows, preds_dir)
    if not rows_available:
        raise FileNotFoundError(
            f"No predicted masks found in: {preds_dir}. Run test first to generate predictions."
        )

    class_map = read_class_map()

    out_pdf = args.output_pdf or (results_mode_dir / "test_evaluation.pdf")
    out_csv = args.output_csv or (results_mode_dir / "test_evaluation_summary.csv")

    write_summary_csv(rows_available, class_map, preds_dir, out_csv)
    build_pdf(rows_available, class_map, preds_dir, out_pdf, args.samples_per_page)

    print(f"Saved PDF evaluation to: {out_pdf}")
    print(f"Saved per-sample class summary to: {out_csv}")
    print(f"Total test samples in split: {len(rows)}")
    print(f"Total samples rendered (prediction available): {len(rows_available)}")
    print(f"Missing predictions: {len(rows_missing)}")
    if rows_missing:
        missing_ids = [r["sample_id"] for r in rows_missing]
        print("Missing sample_id (first 20): " + ", ".join(missing_ids[:20]))


if __name__ == "__main__":
    main()
