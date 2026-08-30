# Classical Detection Pipeline

Unsupervised segmentation baseline for SAR oil slick candidate detection —
benchmark_id `BASELINE_CLASSICAL`, using **Felzenszwalb graph-based
segmentation** as its method. Independent of the deep learning `pipeline_ml`
— requires only the GeoTIFF tiles and the packages in `requirements.txt`.

## benchmark_id vs. method

`config.json`'s `algorithms` block keys are **benchmark IDs**, each naming
the actual **method** it runs (looked up in `scripts/segment.py:ALGO_FUNCS`)
plus that method's parameters — the same split the ML pipeline uses between
`benchmark_id` and `model_name` in `benchmark/BENCHMARK_TABLE.csv`. Today
there is one entry:

| benchmark_id | method | 
|---|---|
| `BASELINE_CLASSICAL` | `felzenszwalb` |

matching the ML side's single `BASELINE_ML` (method `TinyUNet`) — see
`../pipeline_comparison.csv`.

## Why Felzenszwalb

Seven classical methods were originally implemented and compared against
the ground-truth masks (`IMAGES/LABELS-1D/`): CFAR, Otsu, GMM, SLIC+threshold,
Felzenszwalb, watershed, and mean shift. Felzenszwalb had the best pixel-level
IoU (0.276) and the only reasonably balanced precision/recall (0.42 / 0.44) —
every other method either barely detected anything (CFAR: IoU 0.0, mean
shift: IoU 0.002) or over-flagged most of the image as slick (watershed:
recall 0.99 but precision 0.07). The full comparison is saved at
`results/iou_vs_ground_truth.csv`; rerun `scripts/compare_to_ground_truth.py`
to reproduce it.

## What it does

1. **Segments** each of the 173 GeoTIFF tiles with the configured method(s) to produce candidate dark-region polygons.
2. **Extracts features** from each detected polygon using the same geometric and radiometric descriptors as the reference `CSV/Oil_Stats_*.csv` files.
3. **Saves** results per tile: GeoJSON, Shapefile, and a feature CSV.
4. **Produces** a summary table of detection counts across all tiles.

Oil slicks appear as dark regions — low-backscatter patches in Sentinel-1 SAR.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json` before running:

- `dataset_root`: path to the dataset root (contains `RASTER/`) — see the `dataset_root` value
  already set in `config.json` for the correct default in this checkout; it differs between the
  development environment and the published dataset repo, where this pipeline is nested one level
  inside the dataset root instead of pointing at a sibling directory
- `tiff_dir`: subdirectory containing 16-bit GeoTIFF tiles (default: `RASTER/IMG-TIFF/16-BIT`)
- `results_dir`: output directory (default: `./results`)
- `min_polygon_pixels`: minimum polygon size in pixels (default: 50)
- `algorithms.BASELINE_CLASSICAL`: `method` (`felzenszwalb`, looked up in `segment.py`), plus that method's parameters — `scale` (larger = larger segments), `sigma` (pre-smoothing), `min_size`, `dark_percentile` (segments below this mean-intensity percentile are flagged)

## Usage

Run from `pipeline_classical/`:

```bash
python scripts/run_batch.py
```

Compare against ground truth (pixel IoU/precision/recall):

```bash
python scripts/compare_to_ground_truth.py
```

Compare against the ML pipeline's registered benchmarks (same metrics, side by side):

```bash
python scripts/compare_to_ml.py
```

## Output

```
results/
  BASELINE_CLASSICAL/
    <tile_id>/
      <tile_id>.geojson        — detected candidate polygons (GeoJSON)
      <tile_id>.shp            — detected candidate polygons (Shapefile)
      <tile_id>_features.csv   — feature table (same schema as Oil_Stats_*.csv)
  detection_summary.csv        — per-tile detection counts (benchmark_id, method, n_detected)
  iou_vs_ground_truth.csv      — pixel IoU/precision/recall vs. ground truth (benchmark_id, method)
```

`compare_to_ml.py` writes `../pipeline_comparison.csv` at the repo root instead —
it's a cross-pipeline artifact (classical + ML entries side by side: benchmark_id,
method, iou_slick, pixel_accuracy), not a classical-only result.

## Feature schema

The feature CSV uses the same column names as `CSV/Oil_Stats_*.csv` for comparability,
plus three provenance columns at the front: `TILE_ID`, `ALGORITHM`, `POLY_IDX`
(`ALGORITHM` records the **method**, e.g. `felzenszwalb`, not the benchmark_id).

Key differences from the reference CSV:
- No `CLASSE` column (detected polygons are unlabeled).
- `AREA_KM_DS` / `PERIM_KM_DS` equal `AREA_KM2` / `PERIM_KM` (no source shapefile for detected polygons).

## Notes

- Feature extraction code is a standalone port of `src/lib/FunGetFeat_Geom.py` and
  `src/lib/FunGetFeat_Stat.py` — no dependency on `src/Config.py` or `src/lib/FunPlot.py`.
- `compare_to_ground_truth.py` and `compare_to_ml.py` both compute IoU the same
  way the ML pipeline does (global/micro-averaged confusion matrix across all
  pixels of all tiles, positive class = `mask > 0`), so `iou_slick` and
  `pixel_accuracy` are directly comparable across classical and ML results.
