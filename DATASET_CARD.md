# Dataset Card: Oil Spill Segmentation Dataset (800x600)

## 1. Overview

- Name: Oil Spill Segmentation Dataset (800x600)
- Primary task: binary oil slick segmentation and related detection experiments
- Modality: SAR-derived image and mask triplets
- Sensor: Sentinel-1 (C-band SAR)
- Geographic coverage: Gulf of Mexico — Cantarell region
- Acquisition period: January 2020 – September 2020
- Image dimensions: 800 × 600 pixels per tile
- Number of samples: 173 image-mask triplets
- Classes: 2 — `SEA` (background, pixel value 0), `SLICK` (oil spill or seepage, pixel value 1)
- Repository scope: dataset + both baselines reported in its technical validation (classical and ML)

## 2. Repository Tree (Dirs, Files, and Inventory)

```text
oilspill-detection-dataset/
	README.md
		- Repository-level quick overview.
	DATASET_CARD.md
		- This dataset card.
	LICENSE
		- CC BY 4.0, covers the dataset and both benchmark pipelines below.
	CITATION.cff
		- Machine-readable citation metadata.

	CSV/
		- Tabular descriptors and per-polygon statistics for analysis/modeling.
		Oil_Stats_manual_windows_16bits_800x600.csv
			- 385 rows, 67 columns.
			- Polygon-level features computed from 16-bit imagery.
		Oil_Stats_manual_windows_8bits_800x600.csv
			- 385 rows, 67 columns.
			- Polygon-level features computed from 8-bit imagery.

	IMAGES/
		- Image and mask assets for segmentation triplets.
		IMG-RGB/
			- 173 files (input images).
		LABELS-1D/
			- 173 files (single-channel masks).
		LABELS-RGB/
			- 173 files (RGB masks).

	RASTER/
		- Raster and vector-derived geospatial support artifacts.
		IMG-TIFF/
			8-BIT/
				- 173 GeoTIFF tiles (8-bit, aligned to IMG-RGB).
			16-BIT/
				- 173 GeoTIFF tiles (16-bit, full radiometric range).
		LABELS-VECTOR/
			GEOJSON/
				- 173 per-tile polygon label files.
			KML/
				- 173 per-tile KML label files.
			SHAPEFILE/
				- All_Vectors.*: merged shapefile of all polygons.
				- Individual_Vectors.zip: per-tile shapefiles.

	splits/
		- Train/validation/test manifests consumed by training and evaluation scripts.
		train.csv
			- 108 data rows (grouped by source image).
		val.csv
			- 43 data rows (grouped by source image).
		test.csv
			- 22 data rows (grouped by source image).

	pipeline_ml/
		- TinyUNet baseline: training, validation, test, benchmark registration, and reporting utilities.

	pipeline_classical/
		- Felzenszwalb baseline: unsupervised segmentation scripts and registered results
		  (detection_summary.csv, iou_vs_ground_truth.csv).

	pipeline_comparison.csv
		- Classical vs. ML side by side, same metrics (iou_slick, pixel_accuracy) —
		  produced by pipeline_classical/scripts/compare_to_ml.py.
```

## 3. Split Manifest Schema

Each split CSV uses the same schema:

- `sample_id`: unique sample identifier.
- `image_path`: relative path to image input.
- `mask_path`: relative path to segmentation mask.
- `classe`: primary class label.
- `subclasse`: subclass label (currently unpopulated in this package version).

## 4. Feature Table Schema (High Level)

The 8-bit and 16-bit feature tables share the same 67-column schema, including:

- Sample identity and provenance fields (for example image name, polygon indexes, sensor/acquisition context).
- Window and tiling metadata (tile size, panel position, padding, composite flags).
- Geometric descriptors (area, perimeter, complexity, shape metrics, Hu moments, circularity).
- Radiometric statistics for foreground/background regions (`FG_*`, `BG_*`, and `FG_BG_*` groups).
- Target label column (`CLASSE`).

## 5. Naming and Pairing Conventions

- Each `sample_id` should map to one image-mask pair.
- Files are named using the TILE convention: `IMG_XX_TILE_YYY` (e.g., `IMG_01_TILE_001.png`).
- Image and mask paths in split manifests are expected to be valid relative paths within this repository.
- Triplet integrity requires consistent pairing across `IMG-RGB`, `LABELS-1D`, and `LABELS-RGB`.
- Mask pixel values in `LABELS-1D/`: `0 = SEA` (background), `1 = SLICK` (OIL SPILL and SEEPAGE combined).
- `LABELS-RGB/` files use color encoding for visualization only; the ML pipeline consumes `LABELS-1D/`.

## 6. Quality and Validation Notes

- Splits are grouped by source image: all tiles from the same acquisition are kept in the same split.
- Both baselines report metrics identically (global/micro confusion matrix over all pixels of all
  tiles, positive class = `mask > 0`), so `iou_slick`/`pixel_accuracy` are directly comparable between
  `pipeline_comparison.csv` entries.

## 7. Known Limitations

- `classe` and `subclasse` fields in the split manifests are not populated in this version; use the CSV feature tables for polygon-level class labels instead.
- Class imbalance: OIL SPILL (327 polygons) vs. SEEPAGE SLICK (58 polygons) in the source vector layer.
- The `RESULT_BASENAME`/`COMPOSITE_BASENAME` columns in the CSV feature tables retain the internal
  window-generation identifier (`IMG_XX_WIN_YYY`) rather than the public `TILE` naming used by the
  actual files — they are provenance metadata, not file paths, so this does not affect loading images
  via `splits/*.csv` (which already use the `TILE` naming), but don't use those two columns to look up files.

## 8. Intended Use and Compliance

This dataset is intended for research and benchmarking in remote sensing and oil spill analysis. Users are responsible for validating legal, regulatory, and policy constraints for downstream use.

## 9. Citation

> Citation and DOI will be added after publication. If you use this dataset before
> a formal citation is available, please acknowledge INPE (Instituto Nacional de
> Pesquisas Espaciais — Applied Research and Technology Division) and link to the
> repository. See `CITATION.cff` for machine-readable metadata.
