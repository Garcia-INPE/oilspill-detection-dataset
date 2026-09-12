"""
Full version of plot_phase1_vs_ml_comparison.py: instead of 4 hand-picked
tiles, renders RGB / ground truth / ML (TinyUNet) / hybrid (Felzenszwalb +
RF, tuned threshold) for all 22 test tiles, paginated into a PDF, plus a
per-tile IoU summary CSV.

Reuses the same already-saved artifacts as plot_phase1_vs_ml_comparison.py
(see that script's docstring) — no re-inference, no resegmentation.

Usage (from pipeline_hybrid/, after phase1_candidate_classifier.py has run):
  python scripts/plot_phase1_vs_ml_all_tests.py
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from PIL import Image
from rasterio.features import rasterize

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"
_TILES_PER_PAGE = 3

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

    tile_ids = sorted(split_df.index)
    summary_rows = []

    out_pdf = out_dir / "fig7_ml_vs_hybrid_all_test_tiles.pdf"
    pages = (len(tile_ids) + _TILES_PER_PAGE - 1) // _TILES_PER_PAGE

    with PdfPages(out_pdf) as pdf:
        for page in range(pages):
            page_tiles = tile_ids[page * _TILES_PER_PAGE:(page + 1) * _TILES_PER_PAGE]
            fig, axes = plt.subplots(nrows=len(page_tiles), ncols=4,
                                      figsize=(15, 3.6 * len(page_tiles)), squeeze=False)
            fig.suptitle(f"ML vs. hybrid — all test tiles (page {page + 1}/{pages})",
                         fontsize=13, y=0.995)

            for row_i, tile_id in enumerate(page_tiles):
                row = split_df.loc[tile_id]
                rgb = np.asarray(Image.open(dataset_root / row["image_path"]).convert("RGB"))
                gt = np.asarray(Image.open(dataset_root / row["mask_path"])) > 0

                with rasterio.open(tiff_dir / f"{tile_id}.tiff") as src:
                    transform, shape = src.transform, src.shape

                ml_pred = np.asarray(Image.open(ml_pred_dir / f"{tile_id}_pred_mask.png")) > 0
                hybrid_pred = _hybrid_mask(tile_id, preds_df, classical_results_dir, shape, transform)

                iou_ml = _iou(ml_pred, gt)
                iou_hybrid = _iou(hybrid_pred, gt)
                summary_rows.append({
                    "tile_id": tile_id, "gt_positive_px": int(gt.sum()),
                    "iou_ml": iou_ml, "iou_hybrid": iou_hybrid,
                    "iou_delta_ml_minus_hybrid": iou_ml - iou_hybrid,
                })

                panels = [
                    (rgb, f"{tile_id}\nRGB"),
                    (gt, "Ground truth"),
                    (ml_pred, f"ML (TinyUNet)\nIoU = {iou_ml:.3f}"),
                    (hybrid_pred, f"Hybrid (Felz.+RF)\nIoU = {iou_hybrid:.3f}"),
                ]
                for col_i, (arr, title) in enumerate(panels):
                    ax = axes[row_i, col_i]
                    if col_i == 0:
                        ax.imshow(arr)
                    else:
                        ax.imshow(arr, cmap=_MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(title, fontsize=10)
                    ax.axis("off")

            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    summary_df = pd.DataFrame(summary_rows).sort_values("tile_id").reset_index(drop=True)
    summary_csv = results_dir / "phase1_test_per_tile_iou.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[DONE] PDF -> {out_pdf} ({pages} pages, {len(tile_ids)} tiles)")
    print(f"[DONE] Per-tile IoU -> {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
