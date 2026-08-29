# Benchmark Protocol

This protocol standardizes benchmark submissions to keep comparisons fair and reproducible.

## Task Definition
- Task: binary oil slick detection (not configurable for other tasks).
  - Classes: `0=SEA`, `1=SLICK` where `SLICK = OIL SPILL + SEEPAGE`
  - Primary metric: `iou_slick`

## Test Inputs
- Test split manifest: `../splits/test.csv` (relative to `pipeline_ml/`).
- Runtime paths loaded from `scripts/pipeline_ml_config.json`.
- Mask pixel values: `0 = SEA` (background), `1 = SLICK`.

## Primary Metrics
- Primary benchmark score: IoU for class `SLICK` (`iou_slick`).
- Auxiliary metrics:
  - `mean_iou_all` (includes class 0)
  - `pixel_accuracy`
  - `loss`

## Prediction Format (external models)
- One PNG mask per `sample_id` from test split.
- File naming: `<sample_id>_pred_mask.png`
- Resolution must match ground truth mask resolution.
- Pixel values must be class indices in `{0,1}`.

## Visual Evaluation Standard
- Generate side-by-side pages with RGB, ground truth and prediction:
  - `python scripts/visualize_test_evaluation.py --samples-per-page 2`
- To inspect a registered benchmark tag:
  - `python scripts/visualize_test_evaluation.py --benchmark-id BASELINE_ML --samples-per-page 2`
- Output artifacts:
  - `scripts/results/test_reference/test_evaluation.pdf`
  - `scripts/results/test_reference/test_evaluation_summary.csv`
- The summary CSV is path-free and stores only `sample_id`, `gt_*` and `pred_*` class fields.

## Benchmark Table
- CSV table path: `benchmark/BENCHMARK_TABLE.csv`
- Register baseline or new run (visual evaluation + CSV registration in one command):
  - `python scripts/register_benchmark.py --benchmark-id BASELINE_ML`
- The table stores benchmark metadata and metrics only (no artifact paths).
- Artifact locations are deterministic and derived from `<benchmark_id>` under `scripts/results/*_reference/`.
- `--benchmark-id` is required; must be explicitly named to keep experiments traceable.

## Naming Convention
- Benchmark artifacts are named with `<benchmark_id>_` prefix, for example:
  - `BASELINE_ML_metrics_test.json`
  - `BASELINE_ML_test_samples.csv`
  - `BASELINE_ML_test_evaluation.pdf`
  - `BASELINE_ML_test_evaluation_summary.csv`
- Benchmark prediction masks are moved under a tag-named subdirectory (no duplication in root):
  - `scripts/results/test_reference/predictions/<benchmark_id>/`
- After benchmark registration, test artifacts are kept as tag files only; untagged duplicates are not retained.

## Scope
- Registration captures test artifacts only (`metrics_test`, `test_samples`, visual PDF, visual summary CSV).
- If train/validation reference artifacts exist alongside test artifacts, they are also renamed to `<benchmark_id>_*` to prevent duplicate canonical copies.
- The benchmark table stores metadata and metrics only; no file paths.
