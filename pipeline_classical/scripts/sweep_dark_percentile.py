"""
Sensitivity sweep for felzenszwalb's `dark_percentile` parameter (config.json,
currently 30): the percentile cutoff on segment-mean intensity used to decide
which Felzenszwalb segments count as "candidate oil" (see segment.py).

Key efficiency trick: the Felzenszwalb segmentation itself (`scale`, `sigma`,
`min_size`) does NOT depend on dark_percentile -- only the final threshold
that picks which already-computed segments are "dark enough" does. So this
script computes the segmentation once per tile and re-thresholds it for every
percentile in the sweep, instead of re-running felzenszwalb per value.

Small-polygon filtering (min_polygon_pixels, matching segment.py's
mask_to_geodataframe) is applied via connected-component pixel counts
(scipy.ndimage.label, 4-connectivity to match rasterio.features.shapes'
default) rather than a full vectorize/rasterize round trip -- equivalent
result, much cheaper across 13 percentiles x 173 tiles.

Evaluated with the same global/micro-averaged pixel confusion matrix as
compare_to_ground_truth.py, so dark_percentile=30's numbers here should
closely reproduce the registered BASELINE_CLASSICAL row in
results/iou_vs_ground_truth.csv (sanity check, printed at the end).

Usage (from pipeline_classical/):
  python scripts/sweep_dark_percentile.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from segment import _normalize  # noqa: E402

_EPS = 1e-9
_PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
_STRUCTURE = ndimage.generate_binary_structure(2, 1)  # 4-connectivity


def _load_config() -> dict:
    with open(_SCRIPTS_DIR.parent / "config.json") as f:
        return json.load(f)


def _ground_truth_mask(labels_1d_dir: Path, tile_id: str) -> np.ndarray:
    arr = np.array(Image.open(labels_1d_dir / f"{tile_id}_1D.png"))
    return arr > 0


def _dark_mask_for_percentile(segments: np.ndarray, seg_ids: np.ndarray,
                               seg_means: np.ndarray, dark_pct: float,
                               min_pixels: int) -> np.ndarray:
    threshold = np.percentile(seg_means, dark_pct)
    dark_mask = np.zeros(segments.shape, dtype=bool)
    for sid, mean_val in zip(seg_ids, seg_means):
        if mean_val < threshold:
            dark_mask[segments == sid] = True

    labeled, n_components = ndimage.label(dark_mask, structure=_STRUCTURE)
    if n_components == 0:
        return dark_mask
    sizes = ndimage.sum(dark_mask, labeled, index=np.arange(1, n_components + 1))
    small_labels = np.flatnonzero(sizes < min_pixels) + 1
    if len(small_labels):
        dark_mask[np.isin(labeled, small_labels)] = False
    return dark_mask


def main() -> None:
    from skimage.segmentation import felzenszwalb

    cfg = _load_config()
    config_dir = _SCRIPTS_DIR.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    tiff_dir = dataset_root / cfg["tiff_dir"]
    labels_1d_dir = dataset_root / "IMAGES" / "LABELS-1D"
    results_dir = (config_dir / cfg["results_dir"]).resolve()
    min_pixels = int(cfg.get("min_polygon_pixels", 100))

    algo_cfg = cfg["algorithms"]["BASELINE_CLASSICAL"]
    scale = float(algo_cfg.get("scale", 100))
    sigma = float(algo_cfg.get("sigma", 0.5))
    min_size = int(algo_cfg.get("min_size", 50))

    tiff_files = sorted(tiff_dir.glob("*.tiff"))
    confusion = {p: np.zeros(4, dtype=np.int64) for p in _PERCENTILES}  # tp,fp,fn,tn

    print(f"[INFO] Tiles: {len(tiff_files)} | scale={scale} sigma={sigma} "
          f"min_size={min_size} min_polygon_pixels={min_pixels}")
    print(f"[INFO] dark_percentile grid: {_PERCENTILES}")

    for tiff_path in tqdm(tiff_files, desc="Tiles", unit="tile"):
        tile_id = tiff_path.stem
        with rasterio.open(tiff_path) as src:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                band = src.read(1).astype(np.float32)

        gt = _ground_truth_mask(labels_1d_dir, tile_id)

        b_norm = _normalize(band)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            segments = felzenszwalb(b_norm, scale=scale, sigma=sigma,
                                     min_size=min_size, channel_axis=None)
        seg_ids = np.unique(segments)
        seg_means = np.array([b_norm[segments == sid].mean() for sid in seg_ids])

        for dark_pct in _PERCENTILES:
            pred = _dark_mask_for_percentile(segments, seg_ids, seg_means, dark_pct, min_pixels)
            tp = int(np.logical_and(pred, gt).sum())
            fp = int(np.logical_and(pred, ~gt).sum())
            fn = int(np.logical_and(~pred, gt).sum())
            tn = int(np.logical_and(~pred, ~gt).sum())
            confusion[dark_pct] += np.array([tp, fp, fn, tn], dtype=np.int64)

    rows = []
    for dark_pct in _PERCENTILES:
        tp, fp, fn, tn = confusion[dark_pct]
        rows.append({
            "dark_percentile": dark_pct,
            "iou_slick": tp / (tp + fp + fn + _EPS),
            "precision": tp / (tp + fp + _EPS),
            "recall": tp / (tp + fn + _EPS),
            "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + _EPS),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    df = pd.DataFrame(rows)
    out_path = results_dir / "dark_percentile_sweep.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[DONE] Saved -> {out_path}\n")
    print(df.to_string(index=False))

    current = df[df["dark_percentile"] == 30].iloc[0]
    print(f"\n[INFO] Sanity check, dark_percentile=30 vs. registered BASELINE_CLASSICAL "
          f"(results/iou_vs_ground_truth.csv, iou_slick=0.2759): "
          f"iou_slick={current['iou_slick']:.4f}")
    best = df.loc[df["iou_slick"].idxmax()]
    print(f"[INFO] Best in sweep: dark_percentile={int(best['dark_percentile'])} "
          f"(iou_slick={best['iou_slick']:.4f})")


if __name__ == "__main__":
    main()
