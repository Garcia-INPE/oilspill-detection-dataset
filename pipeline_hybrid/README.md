# Hybrid Detection Pipeline

Third approach, distinct from `pipeline_classical` (unsupervised, no labels
used) and `pipeline_ml` (deep learning, end-to-end): classical candidate
generation + a supervised classifier trained on hand-crafted geometric/
radiometric features (the same ones `pipeline_classical/scripts/features.py`
already computes). **Registered benchmark_id `BASELINE_HYBRID`** (method
`felzenszwalb+rf`, adopted 2026-09-11 — see `config.json`'s `benchmark_id`/
`benchmark_method`) — not yet synced to the public dataset repo (the sync
logic exists in `tools/sync_to_dataset_repo.sh` but hasn't been run). Status:
**Phase 0 and Phase 1 done**, see below.

## Why

`pipeline_classical`'s Felzenszwalb detector flags any sufficiently dark
segment as a candidate — it has no notion of whether a segment's statistical
profile actually resembles known oil (precision 0.576, see
`pipeline_classical/results/iou_vs_ground_truth.csv`). The idea: keep
Felzenszwalb (or similar) as a cheap unsupervised *candidate proposer*, then
add a supervised classifier on top of each candidate's features to reject
the ones that don't look like real oil.

## Phase 0 — feasibility check (done)

Before building a full candidate-generation + ground-truth-overlap-labeling
pipeline (expensive), a cheap question first: **are a polygon's own
foreground (oil) pixel statistics already separable from its background
(sea) pixel statistics**, using only what's already in
`CSV/Oil_Stats_manual_windows_16bits_800x600.csv`?

Every row in that CSV is one ground-truth oil polygon, and already carries
two symmetric radiometric profiles for the same tile region (computed by
`pipeline_classical/scripts/features.py:get_feat_stat`):
- `FG_*` — pixel statistics inside the polygon (oil)
- `BG_*` — pixel statistics of the surrounding sea (bbox minus all polygons)

`scripts/phase0_fg_bg_feasibility.py` turns each of the 385 rows into two
training instances — `FG_*` → label 1, `BG_*` → label 0 — using the 7 fields
that exist symmetrically on both sides: `STD, VAR, MIN, MAX, MEAN, MEDIAN,
VAR_COEF`. **Shape/geometric features are deliberately excluded** — `BG` has
no polygon of its own, so there's no shape-equivalent to pair against `FG`'s
shape features. This also directly tests the hypothesis that slick shape is
too variable to be a reliable signal, by construction: shape isn't in the
input at all here.

770 samples (385 FG + 385 BG), zero new candidate generation, zero
ground-truth-overlap labeling — reused entirely from the existing dataset.
Evaluated with `GroupKFold` (5 folds) grouped by `IMG_FNAME` (source scene),
so a FG/BG pair from the same tile never splits across train and test.

### Results (16-bit only, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression on `MEAN` alone (naive baseline) | 0.795 | 0.870 | 0.771 | 0.794 | 0.875 |
| Random Forest, all 7 fields | 0.812 | 0.828 | 0.834 | 0.821 | **0.904** |

Full per-fold numbers: `results/baseline_mean_only_metrics.csv`,
`results/rf_cv_metrics.csv`. Confusion matrix: `results/rf_confusion_matrix.csv`.

**Feature importance** (Random Forest, fit on all data —
`results/rf_feature_importances.csv`):

```
MAX         0.279
MEAN        0.196
MEDIAN      0.163
STD         0.108
VAR         0.092
MIN         0.086
VAR_COEF    0.075
```

### Figures

`scripts/plot_phase0_results.py` generates 4 charts from the results above
into `results/figures/` (feature importance, model comparison, FG-vs-BG
intensity distributions, confusion matrix) — see
`results/figures/ABOUT_CHARTS.md` for a walkthrough of each one.

### Reading

- FG vs. BG radiometric profiles **are separable** (ROC-AUC 0.90) — the idea
  has legs, worth moving to Phase 1.
- The Random Forest beats the naive MEAN-only threshold, but by a modest
  margin (+2.9pp ROC-AUC, +2.7pp F1) — most of the separating power is
  already in raw intensity, not in a sophisticated combination of features.
  `MAX` (not `MEAN`) is the single most important field.
- Fold-to-fold variance is large (fold 3: accuracy 0.65 vs. fold 0: 0.90) —
  expected with only 24 scene-groups; not yet a stable estimate.
- Known limitation of this specific test: `BG_*` is sampled from the sea
  *immediately around a confirmed real slick*. Whether that matches the
  local background of an arbitrary Felzenszwalb false positive elsewhere in
  a tile is untested here — that's exactly what Phase 1 checks.

## Phase 1 — done (2026-09-10, re-run 2026-09-11 after dark_percentile 30→15)

Labels Felzenszwalb's own detected candidates (not the manually-digitized
`Oil_Stats` polygons) by spatial IoU against ground truth, trains a
classifier to reject false-positive candidates, and evaluates with the same
pixel-level protocol as the other two baselines.

`scripts/phase1_candidate_classifier.py`:
- Reuses the candidates + features already produced by
  `pipeline_classical/scripts/run_batch.py` (`BASELINE_CLASSICAL`) — no
  resegmentation. Features: the same geometric (area, perimeter, Hu moments,
  circularity, ...) and radiometric (`FG_*`/`BG_*`/`FG_BG_*`) fields
  `pipeline_classical/scripts/features.py` already computes, minus
  identifiers and duplicate DS-columns; the two categorical KS/MW results
  are binary-encoded. As of `dark_percentile=15` (`pipeline_classical/
  README.md`'s "Parameter sensitivity" section), that's **4,384 candidates**
  across 173 tiles (down from 9,242 at the old `dark_percentile=30` — a
  stricter percentile means fewer, more-confident segments get proposed).
- **Labeling**: a candidate is positive (real oil) if its IoU against the
  best-matching ground-truth polygon in the same tile is `>= min_iou_positive`
  (config.json, 0.3). This is still a hard bar even with fewer, tighter
  candidates: Felzenszwalb's segments remain coarse blobs that rarely match
  a hand-digitized slick's shape tightly — only **1.3%–3.1%** of candidates
  qualify as positive per split (up from 0.6%-1.7% at dark_percentile=30,
  since candidates are individually more concentrated now, but still a
  heavily imbalanced problem by construction, not by bug).
- **Split**: uses the dataset's own `splits/{train,val,test}.csv` — the same
  split `BASELINE_ML` was evaluated on — so the headline test-set number is
  directly comparable. Model selection (GroupKFold by scene, 5-fold) runs on
  `train` only; `phase1_cv_metrics.csv` shows strong ranking ability
  (mean ROC-AUC 0.965) despite the extreme imbalance.
- **Threshold calibration**: the default 0.5 decision threshold is far too
  conservative for this imbalance — it raises precision a lot but collapses
  recall and *drops* IoU below the raw Felzenszwalb baseline. The threshold
  is instead swept on `val` (`phase1_threshold_sweep_val.csv`, grid down to
  0.001) to maximize pixel-level IoU, picked at `0.001`, then applied once
  to `test`. (Sanity-checked against the "keep everything" baseline
  restricted to `val` tiles — IoU 0.226 — confirming the classifier still
  adds real signal at this threshold, not just a no-op filter.)

### Results (22 test tiles, pixel-level, same protocol as `compare_to_ground_truth.py`)

| Variant | IoU | Precision | Recall | Pixel accuracy |
|---|---|---|---|---|
| Felzenszwalb, raw (no filter) | 0.374 | 0.548 | 0.541 | 0.990 |
| + RF filter, threshold 0.5 (default) | 0.268 | 0.731 | 0.298 | 0.991 |
| + RF filter, threshold 0.001 (tuned on val) | **0.459** | 0.769 | 0.532 | 0.993 |

Full numbers: `results/phase1_test_pixel_metrics.csv`,
`results/phase1_threshold_sweep_val.csv`,
`results/phase1_val_test_classification_report.csv`,
`results/phase1_feature_importances.csv`,
`results/phase1_candidates_labeled.csv` (full labeled candidate table, for
reuse/debugging), `results/phase1_test_per_tile_iou.csv` (per-tile IoU,
ML vs. hybrid).

### Reading

- Raw Felzenszwalb alone jumped from IoU 0.272 to 0.374 just from the
  `dark_percentile` fix — most of that gain was already "free" before the
  classifier even runs.
- The candidate classifier **still helps on top of that**: IoU goes from
  0.374 (raw) to 0.459 at the calibrated threshold — a ~23% relative gain,
  precision up from 0.548 to 0.769, recall essentially unchanged (0.541 →
  0.532). Combined with the `dark_percentile` fix, the full pipeline (0.459)
  is a ~69% relative gain over the original dark_percentile=30 + no-filter
  starting point (0.272).
- The default 0.5 threshold is still actively harmful (IoU drops to 0.268,
  below raw) — the decision threshold matters more than the model itself
  here. Always calibrate on `val`, never assume 0.5.
- Per-tile comparison against `BASELINE_ML` (`results/
  phase1_test_per_tile_iou.csv`, `results/figures/
  fig7_ml_vs_hybrid_all_test_tiles.pdf`): the hybrid now strictly beats
  `BASELINE_ML` on 9 of 22 test tiles (up from 4 before the
  `dark_percentile` fix), mostly small/thin slicks where TinyUNet predicts
  nothing at all.
- Still well below `BASELINE_ML`'s IoU 0.605 on the same test protocol —
  but the user decided (2026-09-10) to keep maintaining/documenting this
  third pipeline rather than retire it, and formalized it as `BASELINE_HYBRID`
  (2026-09-11) via `results/benchmark_hybrid.csv` +
  `../pipeline_comparison.csv`. Synced to the dataset repo's working tree
  (2026-09-11, local only, not committed/pushed) and written into
  `ARTICLE_DRAFT.txt` (project root) as Baseline 3.

### XGBoost comparison (2026-09-11)

`scripts/phase1_xgboost_experiment.py` — Random Forest was never chosen by
comparison (inherited from the Phase 0 feasibility check), so this checks
whether XGBoost, which often handles severe class imbalance better on
tabular data, does better here. Same candidates/labeling/split/threshold-
calibration procedure as `phase1_candidate_classifier.py`, only the model
swapped (`scale_pos_weight` set from the train split's class ratio).

| Model | Threshold | IoU | Precision | Recall |
|---|---|---|---|---|
| Random Forest (registered) | 0.001 | **0.459** | 0.769 | 0.532 |
| XGBoost | 0.001 | 0.441 | 0.791 | 0.499 |

RF keeps the win, but narrowly. XGBoost's real advantage showed at the
*uncalibrated* default threshold (0.5): IoU 0.425 vs. RF's 0.268 at the
same default — its native imbalance handling makes it far more robust
when nobody bothers to tune the threshold, even though it doesn't win
once both are properly calibrated. Neither model was hyperparameter-tuned
beyond reasonable defaults. `results/phase1_rf_vs_xgb_comparison.csv`,
`results/phase1_xgb_test_pixel_metrics.csv`. **Registered benchmark stays
Random Forest** — this didn't justify switching.

**Not yet tried**: stacking `BASELINE_ML`'s pixel-level predictions into
this classifier (e.g. mean TinyUNet probability inside each candidate
polygon as an extra RF feature) — a plausible next step, noted in
`ARTICLE_DRAFT.txt`'s Baseline 3 section but not implemented.

## Usage

```bash
pip install -r requirements.txt
python scripts/phase0_fg_bg_feasibility.py
python scripts/phase1_candidate_classifier.py   # after pipeline_classical/scripts/run_batch.py
python scripts/register_benchmark.py            # writes results/benchmark_hybrid.csv (BASELINE_HYBRID)
```

Config (`config.json`): `benchmark_id`, `benchmark_method` (registration),
`dataset_root`, `csv_path` (Phase 0),
`classical_results_dir`, `tiff_dir`, `labels_1d_dir`, `labels_vector_dir`,
`splits_dir`, `min_iou_positive` (Phase 1), `results_dir`, `random_state`,
`n_splits`.
