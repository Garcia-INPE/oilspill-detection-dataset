"""
Side-by-side comparison of classical vs. ML benchmark entries, using the
metrics both sides share: `iou_slick` and `pixel_accuracy`, plus the `method`
each benchmark_id actually applies (e.g. "felzenszwalb" / "TinyUNet").

Both metrics are computed the same way — a global/micro-averaged confusion
matrix across all pixels of all test tiles, positive class = SLICK (mask > 0):
  - Classical:  scripts/compare_to_ground_truth.py -> results/iou_vs_ground_truth.csv
  - ML:         pipeline_ml/scripts/common.py:metrics_from_confusion() -> benchmark/BENCHMARK_TABLE.csv

Run compare_to_ground_truth.py first if results/iou_vs_ground_truth.csv is
missing or stale.

Writes to ../pipeline_comparison.csv (repo root) rather than results/, since
this compares across pipelines rather than reporting a classical-only result.

Usage (from pipeline_classical/):
  python scripts/compare_to_ml.py
"""
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CLASSICAL_IOU_CSV = _SCRIPTS_DIR.parent / "results" / "iou_vs_ground_truth.csv"
_ML_BENCHMARK_CSV = _SCRIPTS_DIR.parent.parent / "pipeline_ml" / "benchmark" / "BENCHMARK_TABLE.csv"
_OUT_CSV = _SCRIPTS_DIR.parent.parent / "pipeline_comparison.csv"


def main() -> None:
    if not _CLASSICAL_IOU_CSV.exists():
        raise FileNotFoundError(
            f"{_CLASSICAL_IOU_CSV} not found — run scripts/compare_to_ground_truth.py first."
        )
    if not _ML_BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"{_ML_BENCHMARK_CSV} not found.")

    classical = pd.read_csv(_CLASSICAL_IOU_CSV)
    ml = pd.read_csv(_ML_BENCHMARK_CSV)

    rows = []
    for _, r in classical.iterrows():
        rows.append({
            "source": "classical",
            "benchmark_id": r["benchmark_id"],
            "method": r["method"],
            "iou_slick": r["iou_slick"],
            "pixel_accuracy": r["pixel_accuracy"],
        })
    for _, r in ml.iterrows():
        rows.append({
            "source": "ml",
            "benchmark_id": r["benchmark_id"],
            "method": r["model_name"],
            "iou_slick": r["iou_slick"],
            "pixel_accuracy": r["pixel_accuracy"],
        })

    df = pd.DataFrame(rows).sort_values("iou_slick", ascending=False).reset_index(drop=True)
    df.to_csv(_OUT_CSV, index=False)

    print(f"\n[DONE] Saved -> {_OUT_CSV}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
