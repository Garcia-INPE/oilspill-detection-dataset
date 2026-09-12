# Oil Spill Segmentation Dataset (800x600)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22214144.svg)](https://doi.org/10.5281/zenodo.22214144)

Sentinel-1 SAR-derived oil slick segmentation dataset covering the Gulf of Mexico
(Cantarell region). 173 image-mask triplets (800 × 600 px) acquired January–September 2020.
Binary segmentation task: `0 = SEA` (background), `1 = SLICK` (oil spill or seepage).

Includes all three baselines reported in the dataset's technical validation: a
classical unsupervised segmentation method, a deep-learning (TinyUNet) model,
and a hybrid method (classical candidate generation + supervised classifier).

Licensed under [CC BY 4.0](LICENSE) — one license for the dataset and all
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
- `pipeline_hybrid/`: hybrid baseline (`BASELINE_HYBRID`) — Felzenszwalb
  candidate generation (reused from `pipeline_classical`) followed by a
  Random Forest classifier on hand-crafted per-candidate features, plus its
  registered results (`results/benchmark_hybrid.csv`). Built to address
  `pipeline_classical`'s low precision — see `pipeline_hybrid/README.md`'s
  "Phase 1" section for the full methodology and results.
- `pipeline_comparison.csv`: all three baselines side by side, same metrics
  (`iou_slick`, `precision`, `recall`, `pixel_accuracy`) — produced by
  `pipeline_classical/scripts/compare_to_ml.py`. Classical is reported at
  two scopes (`full_dataset`, 173 tiles, unsupervised; `test_split`, the
  same 22 tiles as the other two baselines) since it needs no train/test
  split; hybrid and ML are reported on the 22-tile test split only.
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

### Reproducing the hybrid baseline (Felzenszwalb + Random Forest)

`dataset_root` in `pipeline_hybrid/config.json` already points at this
repository's own root (`../`). Requires `pipeline_classical`'s candidates to
exist first (run its `run_batch.py`, above).

```bash
cd pipeline_hybrid
pip install -r requirements.txt
python scripts/phase1_candidate_classifier.py
python scripts/register_benchmark.py
```

See `pipeline_hybrid/README.md` for details, including why this baseline
exists (classical's low precision) and the XGBoost-vs-Random-Forest
comparison that kept Random Forest as the registered model.

## Documentation

- `DATASET_CARD.md`: complete dataset card (schema, inventory, known limitations).
- `pipeline_ml/00-README.md`, `pipeline_ml/BENCHMARK_PROTOCOL.md`: ML workflow and benchmark protocol.
- `pipeline_classical/README.md`: classical baseline details and why Felzenszwalb was selected.
- `pipeline_hybrid/README.md`: hybrid baseline details, rationale, and results.

## Citation

Archived on Zenodo: [10.5281/zenodo.22214144](https://doi.org/10.5281/zenodo.22214144).
See `CITATION.cff` for machine-readable metadata.
