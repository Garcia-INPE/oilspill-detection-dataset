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
IoU and the only reasonably balanced precision/recall — every other method
either barely detected anything (CFAR: IoU 0.0, mean shift: IoU 0.002) or
over-flagged most of the image as slick (watershed: recall 0.99 but
precision 0.07). That comparison used Felzenszwalb's original
`dark_percentile=30` default; current numbers (`dark_percentile=15`, tuned
below) are IoU 0.292, precision 0.576, recall 0.373. The full comparison is
saved at `results/iou_vs_ground_truth.csv`; rerun
`scripts/compare_to_ground_truth.py` to reproduce it.

## Parameter sensitivity: `dark_percentile`

`dark_percentile` (config.json) was originally set to 30 as a carry-over
default, not a value with its own documented sweep — the "why Felzenszwalb"
comparison above tuned which *method* to use, not this method's own knobs.
`scripts/sweep_dark_percentile.py` closes that gap: with `scale`/`sigma`/`min_size`
held fixed, it re-thresholds the same Felzenszwalb segmentation (segmenting
each tile is independent of `dark_percentile` — only the "how dark is dark"
cutoff depends on it, so this is cheap: one segmentation pass per tile,
re-thresholded for every percentile) and scores each value with the same
pixel-level protocol as `compare_to_ground_truth.py`, across all 173 tiles.

![IoU/precision/recall vs. dark_percentile, 173 tiles, current default and empirical best marked](results/figures/dark_percentile_sweep.png)

| dark_percentile | IoU | Precision | Recall |
|---|---|---|---|
| 5 | 0.172 | 0.696 | 0.186 |
| 10 | 0.264 | 0.630 | 0.313 |
| **15 (current default, adopted 2026-09-11)** | **0.292** | 0.574 | 0.373 |
| 20 | 0.291 | 0.516 | 0.400 |
| 25 | 0.287 | 0.466 | 0.426 |
| 30 (previous default) | 0.275 | 0.421 | 0.444 |
| 35 | 0.263 | 0.378 | 0.464 |
| 40 | 0.241 | 0.330 | 0.471 |
| 45 | 0.221 | 0.289 | 0.487 |
| 50 | 0.173 | 0.210 | 0.495 |
| 60 | 0.090 | 0.098 | 0.546 |
| 70 | 0.041 | 0.042 | 0.594 |
| 80 | 0.025 | 0.025 | 0.805 |

Full numbers: `results/dark_percentile_sweep.csv`.

**Reading**: 30 (the original default) was not the optimum — **15 gives IoU
0.292 vs. 0.275, a ~6% relative gain**, by trading some recall (0.373 vs.
0.444) for meaningfully higher precision (0.574 vs. 0.421). The IoU curve
is fairly flat across 10-30 (0.264-0.292) and falls off sharply past 45 as
recall keeps climbing but precision collapses (past dark_percentile=50,
more than half the image gets flagged as candidate slick, and accuracy
craters). Sanity check at the time: the sweep's own dark_percentile=30
point (IoU 0.2754) reproduced the then-registered `BASELINE_CLASSICAL`
value (IoU 0.2759) to within rounding from the connected-component vs.
vector-polygon size filter used by each script — confirmed the two
methodologies agree before adopting the change.

**Adopted 2026-09-11**: `config.json` now uses `dark_percentile=15`.
`results/iou_vs_ground_truth.csv`, `../pipeline_comparison.csv`, and
`pipeline_hybrid`'s Phase 1 (candidates, RF training, threshold
calibration, and the ML-vs-hybrid comparison figures) were all regenerated
against the new candidate set — see `pipeline_hybrid/README.md`'s Phase 1
section for the updated hybrid numbers. The dataset repo's synced copy of
`pipeline_classical` has not been re-synced/republished yet
(`tools/sync_to_dataset_repo.sh` + a new Zenodo version, if desired).

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
