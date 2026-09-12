"""
Side-by-side comparison of all registered benchmark entries (classical, ML,
and — since 2026-09-11 — the hybrid pipeline), using the metrics they all
share: `iou_slick`, `precision`, `recall` and `pixel_accuracy`, plus the
`method` each benchmark_id actually applies (e.g. "felzenszwalb" /
"TinyUNet" / "felzenszwalb+rf"). Name kept as compare_to_ml.py for continuity
with existing docs/history even though it now covers three sources, not two.

All four metrics are computed the same way — a global/micro-averaged
confusion matrix across all pixels of all tiles in scope, positive class =
SLICK (mask > 0):
  - Classical:  scripts/compare_to_ground_truth.py -> results/iou_vs_ground_truth.csv
                (TWO rows: "full_dataset" = all 173 tiles, and "test_split"
                = the 22 held-out test tiles — see that script's docstring
                for why both are kept)
  - ML:         pipeline_ml/scripts/common.py:metrics_from_confusion() -> benchmark/BENCHMARK_TABLE.csv
                (iou_slick/pixel_accuracy) + pipeline_ml/scripts/
                compute_precision_recall.py -> results/test_reference/
                BASELINE_ML_precision_recall_test.csv (precision/recall,
                algebraically derived since test.py never persisted the raw
                confusion matrix — see that script's docstring)
  - Hybrid:     pipeline_hybrid/scripts/phase1_candidate_classifier.py's
                test-split evaluation -> pipeline_hybrid/scripts/
                register_benchmark.py -> results/benchmark_hybrid.csv
                (tuned-threshold variant only, evaluated on the same
                held-out test split as BASELINE_ML)

ML and hybrid are inherently "test_split"-scoped only (they're fit on
train/val, so a full-dataset number would include tiles they were trained
on — not a meaningful held-out metric). Classical is unsupervised and needs
no split to be scored, so "full_dataset" is its original, historically
primary number, kept alongside "test_split" for a fair 3-way comparison
rather than replaced by it — see `scope`/`n_tiles` columns in the output.

Run compare_to_ground_truth.py, compute_precision_recall.py, and
pipeline_hybrid's register_benchmark.py first if their output CSVs are
missing or stale. The hybrid source is optional: if benchmark_hybrid.csv
doesn't exist yet, this script just compares classical + ML as before.

Writes to ../pipeline_comparison.csv (repo root) rather than results/, since
this compares across pipelines rather than reporting a classical-only result.

Usage (from pipeline_classical/):
  python scripts/compare_to_ml.py
"""
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DEV_ROOT = _SCRIPTS_DIR.parent.parent
_CLASSICAL_IOU_CSV = _SCRIPTS_DIR.parent / "results" / "iou_vs_ground_truth.csv"
_ML_BENCHMARK_CSV = _DEV_ROOT / "pipeline_ml" / "benchmark" / "BENCHMARK_TABLE.csv"
_ML_PRECISION_RECALL_CSV = (
    _DEV_ROOT / "pipeline_ml" / "results" / "test_reference"
    / "BASELINE_ML_precision_recall_test.csv"
)
_HYBRID_BENCHMARK_CSV = _DEV_ROOT / "pipeline_hybrid" / "results" / "benchmark_hybrid.csv"
_OUT_CSV = _DEV_ROOT / "pipeline_comparison.csv"
_TEST_SPLIT_N_TILES = 22


def main() -> None:
    if not _CLASSICAL_IOU_CSV.exists():
        raise FileNotFoundError(
            f"{_CLASSICAL_IOU_CSV} not found — run scripts/compare_to_ground_truth.py first."
        )
    if not _ML_BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"{_ML_BENCHMARK_CSV} not found.")
    if not _ML_PRECISION_RECALL_CSV.exists():
        raise FileNotFoundError(
            f"{_ML_PRECISION_RECALL_CSV} not found — run "
            "pipeline_ml/scripts/compute_precision_recall.py first."
        )

    classical = pd.read_csv(_CLASSICAL_IOU_CSV)
    ml = pd.read_csv(_ML_BENCHMARK_CSV)
    ml_precision_recall = pd.read_csv(_ML_PRECISION_RECALL_CSV).set_index("benchmark_id")

    rows = []
    for _, r in classical.iterrows():
        rows.append({
            "source": "classical",
            "benchmark_id": r["benchmark_id"],
            "method": r["method"],
            "scope": r["scope"],
            "n_tiles": int(r["n_tiles"]),
            "iou_slick": r["iou_slick"],
            "precision": r["precision"],
            "recall": r["recall"],
            "pixel_accuracy": r["pixel_accuracy"],
        })
    for _, r in ml.iterrows():
        pr = ml_precision_recall.loc[r["benchmark_id"]]
        rows.append({
            "source": "ml",
            "benchmark_id": r["benchmark_id"],
            "method": r["model_name"],
            "scope": "test_split",
            "n_tiles": _TEST_SPLIT_N_TILES,
            "iou_slick": r["iou_slick"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "pixel_accuracy": r["pixel_accuracy"],
        })
    if _HYBRID_BENCHMARK_CSV.exists():
        hybrid = pd.read_csv(_HYBRID_BENCHMARK_CSV)
        for _, r in hybrid.iterrows():
            rows.append({
                "source": "hybrid",
                "benchmark_id": r["benchmark_id"],
                "method": r["method"],
                "scope": "test_split",
                "n_tiles": _TEST_SPLIT_N_TILES,
                "iou_slick": r["iou_slick"],
                "precision": r["precision"],
                "recall": r["recall"],
                "pixel_accuracy": r["pixel_accuracy"],
            })

    df = pd.DataFrame(rows).sort_values(
        ["scope", "iou_slick"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(_OUT_CSV, index=False)

    print(f"\n[DONE] Saved -> {_OUT_CSV}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
