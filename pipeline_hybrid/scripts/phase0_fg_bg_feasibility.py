"""
Phase 0 feasibility check for the hybrid (classical candidates + supervised
classifier) approach: before building any candidate-generation/labeling
pipeline, test whether a polygon's own foreground (oil) pixel statistics are
already separable from its background (sea) pixel statistics, using only
what's already computed in CSV/Oil_Stats_manual_windows_16bits_800x600.csv.

Every row in that CSV describes one ground-truth oil polygon and already
carries two symmetric radiometric profiles for the same tile region:
  FG_* — pixel statistics inside the polygon (oil)
  BG_* — pixel statistics of the surrounding sea (bbox minus all polygons)
(see pipeline_classical/scripts/features.py:get_feat_stat)

This script turns each row into two training instances — FG_* -> label 1,
BG_* -> label 0 — using only the 7 fields that exist symmetrically on both
sides (STD, VAR, MIN, MAX, MEAN, MEDIAN, VAR_COEF). Shape/geometric features
are deliberately excluded: BG has no polygon of its own to measure, so there
is no shape-equivalent to pair against FG's shape features.

No new candidate generation, no ground-truth overlap labeling — 770 samples
(385 FG + 385 BG) straight from the existing CSV. If this doesn't separate
well, the more expensive full pipeline (label Felzenszwalb's own candidates
by overlap, extract their features, retrain) is not worth building yet.

Evaluation uses GroupKFold on IMG_FNAME (source scene) so a FG/BG pair from
the same tile never splits across train and test.

Usage (from pipeline_hybrid/):
  python scripts/phase0_fg_bg_feasibility.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"

# The 7 radiometric fields computed identically for FG and BG in features.py.
_STAT_FIELDS = ["STD", "VAR", "MIN", "MAX", "MEAN", "MEDIAN", "VAR_COEF"]


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _build_fg_bg_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")

    pos = df[[f"FG_{c}" for c in _STAT_FIELDS]].copy()
    pos.columns = _STAT_FIELDS
    pos["label"] = 1
    pos["group"] = df["IMG_FNAME"]

    neg = df[[f"BG_{c}" for c in _STAT_FIELDS]].copy()
    neg.columns = _STAT_FIELDS
    neg["label"] = 0
    neg["group"] = df["IMG_FNAME"]

    return pd.concat([pos, neg], ignore_index=True)


def _evaluate(model_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              n_splits: int, random_state: int) -> tuple[pd.DataFrame, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    agg_cm = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        model = model_factory(random_state)
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        y_true = y[test_idx]

        rows.append({
            "fold": fold,
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
        })
        agg_cm += confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics_df = pd.DataFrame(rows)
    mean_row = metrics_df.drop(columns=["fold", "n_test"]).mean()
    mean_row["fold"] = "mean"
    mean_row["n_test"] = metrics_df["n_test"].sum()
    metrics_df = pd.concat([metrics_df, mean_row.to_frame().T], ignore_index=True)
    return metrics_df, agg_cm


def main() -> None:
    cfg = _load_config()
    config_dir = _CONFIG_PATH.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    csv_path = dataset_root / cfg["csv_path"]
    results_dir = (config_dir / cfg["results_dir"]).resolve()
    random_state = int(cfg.get("random_state", 42))
    n_splits = int(cfg.get("n_splits", 5))

    results_dir.mkdir(parents=True, exist_ok=True)

    data = _build_fg_bg_dataset(csv_path)
    print(f"[INFO] Samples: {len(data)} ({(data['label'] == 1).sum()} FG / "
          f"{(data['label'] == 0).sum()} BG), groups (scenes): {data['group'].nunique()}")

    X = data[_STAT_FIELDS].to_numpy()
    y = data["label"].to_numpy()
    groups = data["group"].to_numpy()

    # ── Naive baseline: single-feature logistic regression on MEAN alone ──────
    # Context for the RF result — MEAN-only thresholding is roughly what the
    # Felzenszwalb dark-percentile step already does. If RF barely beats this,
    # the extra features/model complexity aren't earning their keep.
    mean_idx = _STAT_FIELDS.index("MEAN")
    baseline_metrics, baseline_cm = _evaluate(
        lambda rs: LogisticRegression(random_state=rs),
        X[:, [mean_idx]], y, groups, n_splits, random_state)
    baseline_metrics.to_csv(results_dir / "baseline_mean_only_metrics.csv", index=False)

    # ── Random Forest on all 7 symmetric radiometric fields ───────────────────
    rf_metrics, rf_cm = _evaluate(
        lambda rs: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs),
        X, y, groups, n_splits, random_state)
    rf_metrics.to_csv(results_dir / "rf_cv_metrics.csv", index=False)

    pd.DataFrame(rf_cm, index=["true_BG", "true_FG"], columns=["pred_BG", "pred_FG"]) \
        .to_csv(results_dir / "rf_confusion_matrix.csv")

    # ── Final model on all data, for feature importance + reuse ───────────────
    final_model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state)
    final_model.fit(X, y)
    importances = pd.Series(final_model.feature_importances_, index=_STAT_FIELDS) \
        .sort_values(ascending=False)
    importances.to_csv(results_dir / "rf_feature_importances.csv", header=["importance"])
    joblib.dump(final_model, results_dir / "rf_model.joblib")

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n[BASELINE] Logistic Regression on MEAN alone:")
    print(baseline_metrics.to_string(index=False))
    print("\n[RANDOM FOREST] All 7 radiometric fields (FG vs BG):")
    print(rf_metrics.to_string(index=False))
    print("\n[RANDOM FOREST] Feature importances (full-data fit):")
    print(importances.to_string())
    print(f"\n[DONE] Results -> {results_dir}")


if __name__ == "__main__":
    main()
