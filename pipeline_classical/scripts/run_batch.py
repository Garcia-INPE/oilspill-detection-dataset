"""
Batch runner: applies all configured segmentation methods to every tile
in the dataset, extracts features from detected candidate polygons, and
saves results per benchmark entry per tile.

Each entry in config.json's "algorithms" block is a benchmark_id (e.g.
"BASELINE_CLASSICAL") mapping to a config that names the actual "method"
(e.g. "felzenszwalb", looked up in segment.ALGO_FUNCS) plus that method's
parameters. This mirrors the ML pipeline's benchmark_id/model_name split
in benchmark/BENCHMARK_TABLE.csv.

Output layout (under results_dir from config.json):
  <benchmark_id>/
    <tile_id>/
      <tile_id>.geojson       — detected polygons (GeoJSON)
      <tile_id>.shp           — detected polygons (Shapefile)
      <tile_id>_features.csv  — feature table for detected polygons
  detection_summary.csv       — per-tile per-benchmark_id detection count

Usage (from pipeline_classical/):
  python scripts/run_batch.py
"""
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm



# ── Locate and import sibling modules ────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from segment import ALGO_FUNCS, mask_to_geodataframe  # noqa: E402
from features import get_feat_geom, get_feat_stat      # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = _SCRIPTS_DIR.parent / "config.json"
    if not config_path.exists():
        print(f"[ERROR] config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Polygon I/O
# ─────────────────────────────────────────────────────────────────────────────

def _save_polygons(gdf: gpd.GeoDataFrame, out_dir: Path, stem: str) -> None:
    """Save a GeoDataFrame as GeoJSON and ESRI Shapefile."""
    if len(gdf) == 0:
        return
    gdf.to_file(str(out_dir / f"{stem}.geojson"), driver="GeoJSON")
    gdf.to_file(str(out_dir / f"{stem}.shp"),     driver="ESRI Shapefile")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(candidate_gdf: gpd.GeoDataFrame,
                      tiff_ds,
                      tile_id: str,
                      method: str) -> pd.DataFrame:
    """
    Extract geometric and radiometric features for every detected polygon.
    Errors per polygon are caught individually so one bad polygon does not
    abort the whole tile.
    """
    rows = []
    for idx in range(len(candidate_gdf)):
        gdf_single = candidate_gdf.iloc[[idx]].copy()
        rec = {
            "TILE_ID":   tile_id,
            "ALGORITHM": method,
            "POLY_IDX":  idx,
        }
        try:
            rec = get_feat_geom(gdf_single, rec)
        except Exception as e:
            rec["_GEOM_ERROR"] = str(e)
        try:
            rec = get_feat_stat(candidate_gdf, gdf_single, rec, tiff_ds)
        except Exception as e:
            rec["_STAT_ERROR"] = str(e)
        rows.append(rec)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    cfg = _load_config()

    config_dir   = _SCRIPTS_DIR.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    tiff_dir     = dataset_root / cfg["tiff_dir"]
    results_dir  = (config_dir / cfg["results_dir"]).resolve()
    min_pixels   = int(cfg.get("min_polygon_pixels", 100))

    results_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(tiff_dir.glob("*.tiff"))
    if not tiff_files:
        print(f"[ERROR] No .tiff files found in {tiff_dir}")
        sys.exit(1)

    print(f"[INFO] Tiles found    : {len(tiff_files)}")
    print(f"[INFO] Benchmark IDs  : {list(cfg['algorithms'].keys())}")
    print(f"[INFO] Results dir    : {results_dir}")

    summary_rows = []

    for tiff_path in tqdm(tiff_files, desc="Tiles", unit="tile"):
        tile_id = tiff_path.stem   # e.g. IMG_01_TILE_001

        with rasterio.open(tiff_path) as src:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                band = src.read(1).astype(np.float32)
            transform = src.transform
            crs       = src.crs

            for benchmark_id, algo_cfg in cfg["algorithms"].items():
                method = algo_cfg.get("method", benchmark_id)

                # ── Segmentation ──────────────────────────────────────────
                try:
                    with warnings.catch_warnings():
                        # Suppress numpy 2.5 shape deprecation and sklearn
                        # joblib/parallel internal notices from third-party libs.
                        warnings.simplefilter("ignore", DeprecationWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        binary_mask = ALGO_FUNCS[method](band, algo_cfg)
                        candidate_gdf = mask_to_geodataframe(
                            binary_mask, transform, crs, min_pixels=min_pixels)
                except Exception as e:
                    tqdm.write(f"[WARN] {tile_id}/{benchmark_id}: segmentation failed — {e}")
                    summary_rows.append({
                        "tile_id": tile_id, "benchmark_id": benchmark_id, "method": method,
                        "n_detected": -1, "status": "segmentation_error"})
                    continue

                # ── Save polygons ─────────────────────────────────────────
                out_dir = results_dir / benchmark_id / tile_id
                out_dir.mkdir(parents=True, exist_ok=True)
                _save_polygons(candidate_gdf, out_dir, tile_id)

                # ── Feature extraction ────────────────────────────────────
                if len(candidate_gdf) > 0:
                    try:
                        feat_df = _extract_features(
                            candidate_gdf, src, tile_id, method)
                        feat_df.to_csv(
                            out_dir / f"{tile_id}_features.csv",
                            sep=";", index=False)
                    except Exception as e:
                        tqdm.write(
                            f"[WARN] {tile_id}/{benchmark_id}: feature extraction failed — {e}")

                summary_rows.append({
                    "tile_id":      tile_id,
                    "benchmark_id": benchmark_id,
                    "method":       method,
                    "n_detected":   len(candidate_gdf),
                    "status":       "ok",
                })

    # ── Summary report ────────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = results_dir / "detection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[DONE] Summary → {summary_path}")

    ok = summary_df[summary_df["status"] == "ok"]
    if not ok.empty:
        pivot = ok.pivot_table(
            index="benchmark_id", values="n_detected",
            aggfunc={"n_detected": ["sum", "mean"]})
        pivot.columns = ["total_detected", "mean_per_tile"]
        pivot["mean_per_tile"] = pivot["mean_per_tile"].round(1)
        print("\nDetection count per benchmark_id (across all tiles):")
        print(pivot.sort_values("total_detected", ascending=False).to_string())


if __name__ == "__main__":
    run()
