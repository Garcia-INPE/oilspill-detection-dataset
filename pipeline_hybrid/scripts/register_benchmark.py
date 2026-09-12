"""
Registers the hybrid pipeline as a formal benchmark entry, in the same
benchmark_id/method schema pipeline_classical's results/iou_vs_ground_truth.csv
uses (benchmark_id, method, iou_slick, precision, recall, pixel_accuracy,
tp, fp, fn, tn) -- so scripts/compare_to_ml.py in pipeline_classical can
read it alongside the classical and ML rows without a special case.

Takes the "felzenszwalb_plus_rf_tuned_on_val" row from
results/phase1_test_pixel_metrics.csv (the tuned-threshold result on the
held-out test split -- the only one of the three Phase 1 variants that's a
fair, non-overfit number, exactly like how BASELINE_ML is scored only on
its own test split). Does not re-run anything: phase1_candidate_classifier.py
must have already produced that file.

Usage (from pipeline_hybrid/, after phase1_candidate_classifier.py has run):
  python scripts/register_benchmark.py
"""
import json
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"
_VARIANT = "felzenszwalb_plus_rf_tuned_on_val"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def main() -> None:
    cfg = _load_config()
    benchmark_id = cfg["benchmark_id"]
    method = cfg["benchmark_method"]
    results_dir = (_CONFIG_PATH.parent / cfg["results_dir"]).resolve()

    metrics_path = results_dir / "phase1_test_pixel_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found -- run scripts/phase1_candidate_classifier.py first."
        )

    df = pd.read_csv(metrics_path)
    row = df[df["variant"] == _VARIANT]
    if row.empty:
        raise ValueError(f"Variant '{_VARIANT}' not found in {metrics_path}")
    row = row.iloc[0]

    out_row = {
        "benchmark_id": benchmark_id,
        "method": method,
        "iou_slick": row["iou_slick"],
        "precision": row["precision"],
        "recall": row["recall"],
        "pixel_accuracy": row["pixel_accuracy"],
        "tp": int(row["tp"]), "fp": int(row["fp"]),
        "fn": int(row["fn"]), "tn": int(row["tn"]),
    }
    out_path = results_dir / "benchmark_hybrid.csv"
    pd.DataFrame([out_row]).to_csv(out_path, index=False)

    print(f"[DONE] Registered {benchmark_id} -> {out_path}")
    print(pd.DataFrame([out_row]).to_string(index=False))


if __name__ == "__main__":
    main()
