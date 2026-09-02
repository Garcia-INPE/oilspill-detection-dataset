# Oil Spill Segmentation Dataset (800x600)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22214145.svg)](https://doi.org/10.5281/zenodo.22214145)

Sentinel-1 SAR-derived oil slick segmentation dataset covering the Gulf of Mexico
(Cantarell region). 173 image-mask triplets (800 × 600 px) acquired January–September 2020.
Binary segmentation task: `0 = SEA` (background), `1 = SLICK` (oil spill or seepage).

Includes both baselines reported in the dataset's technical validation: a
classical unsupervised segmentation method and a deep-learning (TinyUNet) model.

Licensed under [CC BY 4.0](LICENSE) — one license for the dataset and both
benchmark pipelines below.

## Repository contents

- `IMAGES/`: input images and segmentation masks (173 triplets: `IMG-RGB`, `LABELS-1D`, `LABELS-RGB`).
- `CSV/`: tabular feature/statistics tables (8-bit and 16-bit, 385 polygons, 67 columns each).
- `RASTER/`: geospatial support artifacts — GeoTIFF (8/16-bit), GeoJSON, KML, Shapefile.
  Shapefile bundles live in `RASTER/LABELS-VECTOR/SHAPEFILE/`:
  `All_Vectors.*` (merged, all polygons) and `Individual_Vectors.zip` (per-tile).
- `splits/`: train/validation/test manifests (108 / 43 / 22 samples).
- `pipeline_ml/`: TinyUNet baseline — training, evaluation, and benchmark scripts.
- `pipeline_classical/`: Felzenszwalb baseline — unsupervised segmentation scripts
  and registered results (`detection_summary.csv`, `iou_vs_ground_truth.csv`).
  Felzenszwalb was chosen as the best of 7 classical methods benchmarked
  against ground truth, not an arbitrary pick — see `pipeline_classical/README.md`'s
  "Why Felzenszwalb" section.
- `pipeline_comparison.csv`: classical vs. ML side by side, same metrics
  (`iou_slick`, `pixel_accuracy`) — produced by `pipeline_classical/scripts/compare_to_ml.py`.
- `DATASET_CARD.md`: full dataset reference (schema, inventory, limitations).
- `LICENSE`: CC BY 4.0 license.
- `CITATION.cff`: machine-readable citation metadata.

## Getting started

### Loading splits with Python

```python
import pandas as pd
from pathlib import Path
from PIL import Image

dataset_root = Path(".")   # path to the cloned/unzipped repo

train = pd.read_csv(dataset_root / "splits/train.csv")

# Load one sample
row = train.iloc[0]
image = Image.open(dataset_root / row["image_path"])  # 800x600 RGB
mask  = Image.open(dataset_root / row["mask_path"])   # 800x600 single-channel (0=SEA, 1=SLICK)
```

### Reproducing the ML baseline (TinyUNet)

All paths (dataset root, results, benchmark) are resolved exclusively from
`pipeline_ml/scripts/pipeline_ml_config.json`. The defaults work out-of-the-box
when running from a cloned/unzipped copy of this repository — no environment
variables or CLI flags needed. Edit the JSON file only if you move directories.

```bash
cd pipeline_ml
bash 01-install.sh cpu    # or: bash 01-install.sh gpu
bash 02-run_pipeline_ml.sh train
bash 02-run_pipeline_ml.sh validate
bash 02-run_pipeline_ml.sh test
```

See `pipeline_ml/00-README.md` for full pipeline documentation and
`pipeline_ml/BENCHMARK_PROTOCOL.md` for the benchmark submission format.

### Reproducing the classical baseline (Felzenszwalb)

`dataset_root` in `pipeline_classical/config.json` already points at this
repository's own root (`../`).

```bash
cd pipeline_classical
pip install -r requirements.txt
python scripts/run_batch.py
python scripts/compare_to_ground_truth.py
```

See `pipeline_classical/README.md` for details.

## Documentation

- `DATASET_CARD.md`: complete dataset card (schema, inventory, known limitations).
- `pipeline_ml/00-README.md`, `pipeline_ml/BENCHMARK_PROTOCOL.md`: ML workflow and benchmark protocol.
- `pipeline_classical/README.md`: classical baseline details and why Felzenszwalb was selected.

## Citation

Archived on Zenodo: [10.5281/zenodo.22214145](https://doi.org/10.5281/zenodo.22214145).
See `CITATION.cff` for machine-readable metadata.
