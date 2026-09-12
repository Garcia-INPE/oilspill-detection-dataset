"""
Compares each classical benchmark entry's detected polygons against the
published ground-truth masks (IMAGES/LABELS-1D), producing a pixel-level
IoU/precision/recall ranking directly comparable to the ML/hybrid
pipelines' `iou_slick` metric.

Ground truth convention (matches pipeline_ml/scripts/common.py:remap_mask):
  LABELS-1D pixel values are 0 (SEA) / 1 (OIL SPILL) / 2 (SEEPAGE); both 1 and
  2 count as the positive "SLICK" class via `mask > 0`.

Reports TWO scopes per benchmark_id, not one:
  - "full_dataset" (all 173 tiles): the original evaluation, computed before
    train/val/test splits existed for this project (classical is
    unsupervised, it never needed a split to be scored) -- this is the
    first-computed, historically primary number.
  - "test_split" (the 22 held-out test tiles): added once BASELINE_ML (and
    later BASELINE_HYBRID) needed a fair, fit-on-train-only comparison --
    those two can only be scored on their own test split, so this second
    scope is what's actually apples-to-apples across all three pipelines.
Both are kept (not one replacing the other) so the comparison table can be
read either way, with the scope made explicit rather than implied.

Confusion counts are accumulated globally across tiles in each scope
(micro-averaged), matching pipeline_ml/scripts/common.py:metrics_from_confusion.

Usage (from pipeline_classical/):
  python scripts/compare_to_ground_truth.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from PIL import Image
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"
_RESULTS_DIR = _SCRIPTS_DIR.parent / "results"
_EPS = 1e-9


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _ground_truth_mask(labels_1d_dir: Path, tile_id: str) -> np.ndarray:
    path = labels_1d_dir / f"{tile_id}_1D.png"
    arr = np.array(Image.open(path))
    return arr > 0  # 0=SEA; 1 (OIL SPILL) and 2 (SEEPAGE) both count as SLICK


def _prediction_mask(benchmark_dir: Path, tile_id: str, shape, transform) -> np.ndarray:
    geojson_path = benchmark_dir / tile_id / f"{tile_id}.geojson"
    if not geojson_path.exists():
        return np.zeros(shape, dtype=bool)
    with open(geojson_path) as f:
        gj = json.load(f)
    geoms = [feat["geometry"] for feat in gj.get("features", [])]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(
        [(g, 1) for g in geoms],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask > 0


def main() -> None:
    cfg = _load_config()
    dataset_root = (_SCRIPTS_DIR.parent / cfg["dataset_root"]).resolve()
    tiff_dir = dataset_root / cfg["tiff_dir"]
    labels_1d_dir = dataset_root / "IMAGES" / "LABELS-1D"
    test_tile_ids = set(pd.read_csv(dataset_root / "splits" / "test.csv")["sample_id"])

    tiff_files = sorted(tiff_dir.glob("*.tiff"))
    # benchmark_id -> method (e.g. "BASELINE_CLASSICAL" -> "felzenszwalb")
    benchmarks = {
        benchmark_id: algo_cfg.get("method", benchmark_id)
        for benchmark_id, algo_cfg in cfg["algorithms"].items()
    }
    scopes = {"full_dataset": None, "test_split": test_tile_ids}

    # confusion[(benchmark_id, scope)] = [TP, FP, FN, TN]
    confusion = {
        (bid, scope): np.zeros(4, dtype=np.int64)
        for bid in benchmarks for scope in scopes
    }

    for tiff_path in tqdm(tiff_files, desc="Tiles", unit="tile"):
        tile_id = tiff_path.stem
        with rasterio.open(tiff_path) as src:
            transform, shape = src.transform, src.shape

        gt = _ground_truth_mask(labels_1d_dir, tile_id)

        for benchmark_id in benchmarks:
            benchmark_dir = _RESULTS_DIR / benchmark_id
            pred = _prediction_mask(benchmark_dir, tile_id, shape, transform)

            tp = int(np.logical_and(pred, gt).sum())
            fp = int(np.logical_and(pred, ~gt).sum())
            fn = int(np.logical_and(~pred, gt).sum())
            tn = int(np.logical_and(~pred, ~gt).sum())
            counts = np.array([tp, fp, fn, tn], dtype=np.int64)

            for scope, tile_filter in scopes.items():
                if tile_filter is None or tile_id in tile_filter:
                    confusion[(benchmark_id, scope)] += counts

    rows = []
    for benchmark_id, method in benchmarks.items():
        for scope, tile_filter in scopes.items():
            tp, fp, fn, tn = confusion[(benchmark_id, scope)]
            iou_slick = tp / (tp + fp + fn + _EPS)
            precision = tp / (tp + fp + _EPS)
            recall = tp / (tp + fn + _EPS)
            pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + _EPS)
            n_tiles = len(test_tile_ids) if tile_filter is not None else len(tiff_files)
            rows.append({
                "benchmark_id": benchmark_id,
                "method": method,
                "scope": scope,
                "n_tiles": n_tiles,
                "iou_slick": iou_slick,
                "precision": precision,
                "recall": recall,
                "pixel_accuracy": pixel_accuracy,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

    df = pd.DataFrame(rows).sort_values(
        ["scope", "iou_slick"], ascending=[True, False]).reset_index(drop=True)
    out_path = _RESULTS_DIR / "iou_vs_ground_truth.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[DONE] Saved -> {out_path}\n")
    print(df[["benchmark_id", "method", "scope", "n_tiles", "iou_slick", "precision",
              "recall", "pixel_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
