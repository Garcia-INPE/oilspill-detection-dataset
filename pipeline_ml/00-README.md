# ML Pipeline (Train, Validate, Test)

This folder contains scripts for the full oil slick segmentation pipeline: training, validation, testing, and benchmarking.

## Pipeline Overview
- Specific to oil slick detection; not a general-purpose segmentation framework.
	- Classes: `0=SEA`, `1=SLICK` where `SLICK = OIL SPILL + SEEPAGE`
	- Primary metric: `iou_slick`
- All input/output paths resolve from `scripts/pipeline_ml_config.json`.

## What Is Included
- `requirements-common.txt`: shared Python dependencies.
- `requirements-cpu.txt`: CPU profile for torch/torchvision.
- `requirements-gpu.txt`: GPU profile for torch/torchvision.
- `01-install.sh`: package installation.
- `02-run_pipeline_ml.sh`: one command for train/validate/test execution.
- `scripts/train.py`: training pipeline with checkpoint persistence.
- `scripts/validate.py`: validation pipeline using a saved checkpoint.
- `scripts/test.py`: test pipeline with persisted metrics and predicted masks.
- `scripts/common.py`, `scripts/model.py`: source and model utilities.

## Setup
From this directory:

```bash
bash 01-install.sh cpu
```

All path resolution uses a single source of truth — `scripts/pipeline_ml_config.json`.
No CLI path overrides or environment variables are used. The shipped defaults work
out-of-the-box for the standard repository layout:

```json
{
  "dataset_root": "../../",
  "results_dir": "../results",
  "benchmark_dir": "../benchmark"
}
```

Required keys:
- `dataset_root`: directory containing `splits/` — defaults to the repository root (`../../` relative to `scripts/`)
- `results_dir`: output directory for run artifacts — defaults to `results/` (i.e. `pipeline_ml/results/`, `../results` relative to `scripts/`)
- `benchmark_dir`: directory containing `BENCHMARK_TABLE.csv` — defaults to `pipeline_ml/benchmark/`

Relative paths are resolved from the directory that contains `pipeline_ml_config.json`
(i.e. `scripts/`), then normalized to absolute paths. If the file is missing or a
required key is absent, scripts fail immediately with a descriptive error.

For CUDA-enabled environments:

```bash
bash 01-install.sh gpu
```

## Training
```bash
bash 02-run_pipeline_ml.sh train
```

Artifacts are saved to:
- `results/<run-name>/checkpoints/best.pt`
- `results/<run-name>/checkpoints/last.pt`
- `results/<run-name>/history.json`

## Validation
```bash
bash 02-run_pipeline_ml.sh validate
```

Metrics are saved to:
- `results/<run-name>/metrics_val.json`

## Testing
```bash
bash 02-run_pipeline_ml.sh test
```

Test artifacts are saved to:
- `results/<run-name>/metrics_test.json`
- `results/<run-name>/test_samples.csv`
- `results/<run-name>/predictions/*.png`

## Visual Evaluation
Side-by-side PDF (RGB image, ground truth mask, predicted mask) for all test samples:

```bash
python scripts/visualize_test_evaluation.py --samples-per-page 2
```

To render evaluation for a registered benchmark tag:

```bash
python scripts/visualize_test_evaluation.py --benchmark-id BASELINE_ML --samples-per-page 2
```

Outputs:
- `results/test_reference/test_evaluation.pdf`
- `results/test_reference/test_evaluation_summary.csv`

Notes:
- Only existing predicted masks are visualized; missing predictions are not generated automatically.
- The summary CSV stores `sample_id` and class presence fields (`gt_*`, `pred_*`) only — no file paths.

## Notes
- Scripts accept `--device auto|cpu|cuda`; `auto` selects the available device.
- Primary metric is IoU for class 1 (SLICK); class 0 (SEA) is excluded from primary reporting.

## TODO
- Validate the `gpu` installation profile (`bash 01-install.sh gpu`) on a CUDA-enabled environment before final release.

## Benchmarking

See `BENCHMARK_PROTOCOL.md` for the full protocol. Quick reference:

- Benchmark table: `benchmark/BENCHMARK_TABLE.csv`
- Register a run (snapshots results + generates visual evaluation):

```bash
python scripts/register_benchmark.py --benchmark-id BASELINE_ML
```

- Compare two registered runs:

```bash
python scripts/compare_benchmarks.py --benchmark-a BASELINE_ML --benchmark-b EXPERIMENT_01
```

- `--benchmark-id` is required for all benchmark operations.
- Registered artifacts use `<benchmark_id>_` prefix and are immutable after registration.
