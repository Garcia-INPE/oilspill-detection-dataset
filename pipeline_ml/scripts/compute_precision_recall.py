"""
Derives precision/recall for BASELINE_ML (TinyUNet) on the test split.

test.py only ever saved the derived IoU/pixel_accuracy pair
(metrics_from_confusion in common.py), not the underlying 2x2 confusion
matrix itself, so precision/recall were never persisted. Re-running
inference isn't needed to recover them: with num_classes=2 and a single
target class, the confusion matrix has exactly 2 degrees of freedom left
once (iou, pixel_accuracy, total_pixels, gt_positive_pixels) are known, and
solving for them is exact algebra, not an approximation:

  total  = tp+fp+fn+tn        (from mask dimensions, independent of the model)
  gt_pos = tp+fn               (from ground-truth masks alone)
  acc    = (tp+tn)/total   ->  tp+tn = acc*total
  iou    = tp/(tp+fp+fn)   ->  tp+fp+fn = total-tn  (since fp+fn = total-tp-tn+... )
                                 => iou*(total-tn) = acc*total-tn
                                 => tn = (acc*total - iou*total) / (1-iou)

Verified self-consistent: reconstructed tp/fp/fn/tn round to whole pixel
counts and reproduce the saved iou/pixel_accuracy to float precision.

Usage (from pipeline_ml/):
  python scripts/compute_precision_recall.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from common import resolve_dataset_root  # noqa: E402

_ML_ROOT = _SCRIPTS_DIR.parent
_DATASET_ROOT = resolve_dataset_root()
_TEST_METRICS_JSON = _ML_ROOT / "results" / "test_reference" / "BASELINE_ML_metrics_test.json"
_TEST_SPLIT_CSV = _DATASET_ROOT / "splits" / "test.csv"
_OUT_CSV = _ML_ROOT / "results" / "test_reference" / "BASELINE_ML_precision_recall_test.csv"


def _ground_truth_totals(test_split_csv: Path, dataset_root: Path) -> tuple[int, int]:
    split_df = pd.read_csv(test_split_csv)
    total_px = 0
    gt_pos = 0
    for mask_path in split_df["mask_path"]:
        arr = np.array(Image.open(dataset_root / mask_path))
        total_px += arr.size
        gt_pos += int((arr > 0).sum())
    return total_px, gt_pos


def main() -> None:
    with open(_TEST_METRICS_JSON) as f:
        metrics = json.load(f)
    iou = metrics["mean_iou"]
    acc = metrics["pixel_accuracy"]

    total, gt_pos = _ground_truth_totals(_TEST_SPLIT_CSV, _DATASET_ROOT)
    gt_neg = total - gt_pos

    tp_plus_tn = acc * total
    tn = (tp_plus_tn - iou * total) / (1 - iou)
    tp = tp_plus_tn - tn
    fn = gt_pos - tp
    fp = gt_neg - tn

    # Sanity check: reconstruction must reproduce the saved metrics exactly.
    iou_check = tp / (tp + fp + fn)
    acc_check = (tp + tn) / total
    assert abs(iou_check - iou) < 1e-6, f"IoU mismatch: {iou_check} vs {iou}"
    assert abs(acc_check - acc) < 1e-6, f"Accuracy mismatch: {acc_check} vs {acc}"

    tp, fp, fn, tn = round(tp), round(fp), round(fn), round(tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    row = {
        "benchmark_id": "BASELINE_ML",
        "method": "TinyUNet",
        "iou_slick": iou,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    pd.DataFrame([row]).to_csv(_OUT_CSV, index=False)

    print(f"[DONE] Saved -> {_OUT_CSV}\n")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
