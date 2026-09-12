"""
Visual side-by-side: RGB tile / ground truth / ML (TinyUNet) prediction /
hybrid (Felzenszwalb + RF, threshold tuned on val) prediction, for a
hand-picked set of test tiles spanning the range of outcomes (ML clearly
ahead, a case where the hybrid actually beats ML, a case where both
struggle).

Reuses already-saved artifacts, no re-inference:
  - RGB/GT:     dataset_root/splits/test.csv -> IMAGES/IMG-RGB, IMAGES/LABELS-1D
  - ML pred:    pipeline_ml/results/test_reference/predictions/BASELINE_ML/<tile>_pred_mask.png
  - Hybrid pred: rasterizes the candidates kept at the tuned threshold, from
                 results/phase1_test_candidate_predictions.csv (kept_tuned==1)
                 + pipeline_classical/results/BASELINE_CLASSICAL/<tile>/<tile>.geojson

Usage (from pipeline_hybrid/, after phase1_candidate_classifier.py has run):
  python scripts/plot_phase1_vs_ml_comparison.py
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from PIL import Image
from rasterio.features import rasterize

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"

# Hand-picked for spread, from a full per-tile IoU scan of all 22 test tiles:
#   IMG_07_TILE_004 - both catch it, ML clearly ahead (large slick)
#   IMG_17_TILE_006 - both catch it, ML clearly ahead (large slick)
#   IMG_17_TILE_001 - ML misses entirely, hybrid partially catches it
#   IMG_20_TILE_003 - both struggle (hybrid misses, ML barely registers)
_SELECTED_TILES = ["IMG_07_TILE_004", "IMG_17_TILE_006", "IMG_17_TILE_001", "IMG_20_TILE_003"]

_GRID = "#dedcd6"
_TEXT = "#3a3a38"
_OIL = "#eb6834"
_MASK_CMAP = ListedColormap(["#000000", _OIL])

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": _TEXT,
    "font.size": 10,
})


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    eps = 1e-9
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float(tp / (tp + fp + fn + eps))


def _hybrid_mask(tile_id: str, preds_df: pd.DataFrame, classical_results_dir: Path,
                  shape, transform) -> np.ndarray:
    sub = preds_df[preds_df["TILE_ID"] == tile_id]
    keep_idx = set(sub.loc[sub["kept_tuned"] == 1, "POLY_IDX"])
    if not keep_idx:
        return np.zeros(shape, dtype=bool)
    gdf = gpd.read_file(classical_results_dir / "BASELINE_CLASSICAL" / tile_id / f"{tile_id}.geojson")
    geoms = [gdf.geometry.iloc[i] for i in keep_idx if i < len(gdf)]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    mask = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform,
                      fill=0, dtype="uint8")
    return mask > 0


def main() -> None:
    cfg = _load_config()
    config_dir = _CONFIG_PATH.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    classical_results_dir = (config_dir / cfg["classical_results_dir"]).resolve()
    tiff_dir = dataset_root / cfg["tiff_dir"]
    results_dir = (config_dir / cfg["results_dir"]).resolve()
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_pred_dir = config_dir.parent / "pipeline_ml" / "results" / "test_reference" / "predictions" / "BASELINE_ML"
    split_df = pd.read_csv(dataset_root / "splits" / "test.csv").set_index("sample_id")
    preds_df = pd.read_csv(results_dir / "phase1_test_candidate_predictions.csv")

    n = len(_SELECTED_TILES)
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(15, 3.6 * n))

    for row_i, tile_id in enumerate(_SELECTED_TILES):
        row = split_df.loc[tile_id]
        rgb = np.asarray(Image.open(dataset_root / row["image_path"]).convert("RGB"))
        gt = np.asarray(Image.open(dataset_root / row["mask_path"])) > 0

        with rasterio.open(tiff_dir / f"{tile_id}.tiff") as src:
            transform, shape = src.transform, src.shape

        ml_pred = np.asarray(Image.open(ml_pred_dir / f"{tile_id}_pred_mask.png")) > 0
        hybrid_pred = _hybrid_mask(tile_id, preds_df, classical_results_dir, shape, transform)

        iou_ml = _iou(ml_pred, gt)
        iou_hybrid = _iou(hybrid_pred, gt)

        panels = [
            (rgb, f"{tile_id}\nRGB", None),
            (gt, "Ground truth", None),
            (ml_pred, f"ML (TinyUNet)\nIoU = {iou_ml:.3f}", None),
            (hybrid_pred, f"Hybrid (Felz.+RF)\nIoU = {iou_hybrid:.3f}", None),
        ]
        for col_i, (arr, title, _) in enumerate(panels):
            ax = axes[row_i, col_i]
            if col_i == 0:
                ax.imshow(arr)
            else:
                ax.imshow(arr, cmap=_MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle("ML vs. hybrid (Felzenszwalb + RF, tuned threshold) on selected test tiles",
                  fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path = out_dir / "fig6_ml_vs_hybrid_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] Figure -> {out_path}")


if __name__ == "__main__":
    main()
