"""
Experiment: does XGBoost beat Random Forest as Phase 1's candidate
classifier? Random Forest was never chosen by comparison -- it was
inherited from the Phase 0 feasibility check that preceded Phase 1 (see
phase1_candidate_classifier.py's docstring) and kept for consistency. Given
the extreme class imbalance here (1.3%-3.1% positive candidates per split),
gradient boosting's native imbalance handling (scale_pos_weight) is a
plausible improvement, per the literature on tabular classification under
skewed classes.

Reuses phase1_candidate_classifier.py's labeling/dataset-construction/
pixel-evaluation machinery as-is (same candidates, same IoU>=0.3 labeling,
same train/val/test split, same threshold-calibration-on-val procedure) --
only the model class changes, so the comparison is apples-to-apples. Writes
to separate phase1_xgb_*.csv files; does NOT touch the registered
BASELINE_HYBRID artifacts (results/benchmark_hybrid.csv etc.), which stay
the Random Forest result until/unless this experiment justifies switching.

Usage (from pipeline_hybrid/, after phase1_candidate_classifier.py has run
at least once so pipeline_classical's candidates are labeled the same way):
  python scripts/phase1_xgboost_experiment.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from phase1_candidate_classifier import (  # noqa: E402
    _build_dataset,
    _build_tile_cache,
    _feature_matrix,
    _keep_idx_at_threshold,
    _load_config,
    _pixel_metrics_from_cache,
    _sweep_threshold,
    _CONFIG_PATH,
)


def _make_model(y_train: np.ndarray, random_state: int) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     n_splits: int, random_state: int) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        model = _make_model(y[train_idx], random_state)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[test_idx], y_pred),
            "precision": precision_score(y[test_idx], y_pred, zero_division=0),
            "recall": recall_score(y[test_idx], y_pred, zero_division=0),
            "f1": f1_score(y[test_idx], y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y[test_idx], y_proba),
        })
    cv_df = pd.DataFrame(rows)
    mean_row = cv_df.drop(columns="fold").mean().to_dict()
    mean_row["fold"] = "mean"
    return pd.concat([cv_df, pd.DataFrame([mean_row])], ignore_index=True)


def main() -> None:
    cfg = _load_config()
    config_dir = _CONFIG_PATH.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    classical_results_dir = (config_dir / cfg["classical_results_dir"]).resolve()
    tiff_dir = dataset_root / cfg["tiff_dir"]
    labels_1d_dir = dataset_root / cfg["labels_1d_dir"]
    results_dir = (config_dir / cfg["results_dir"]).resolve()
    random_state = int(cfg.get("random_state", 42))
    n_splits = int(cfg.get("n_splits", 5))

    print("[INFO] Building dataset (same labeling as phase1_candidate_classifier.py) ...")
    full_df = _build_dataset(cfg, config_dir)
    full_df = full_df[full_df["split"] != "unknown"].reset_index(drop=True)
    print(f"[INFO] Total candidates: {len(full_df)}")

    X_all, feature_names = _feature_matrix(full_df)
    y_all = full_df["label"].to_numpy()

    train_mask = (full_df["split"] == "train").to_numpy()
    val_mask = (full_df["split"] == "val").to_numpy()
    test_mask = (full_df["split"] == "test").to_numpy()

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    groups_train = full_df.loc[train_mask, "scene"].to_numpy()

    # ── Cross-validated feasibility check on the train split ────────────────
    print("[INFO] Running GroupKFold cross-validation (XGBoost) on the train split ...")
    cv_df = _cross_validate(X_train, y_train, groups_train, n_splits, random_state)
    cv_df.to_csv(results_dir / "phase1_xgb_cv_metrics.csv", index=False)
    print(cv_df.to_string(index=False))

    # ── Final model: fit on full train split ─────────────────────────────────
    model = _make_model(y_train, random_state)
    model.fit(X_train, y_train)

    importances = pd.DataFrame({
        "feature": feature_names, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(results_dir / "phase1_xgb_feature_importances.csv", index=False)

    # ── Threshold calibration on the val split (maximize pixel-level IoU) ────
    val_df = full_df[val_mask].copy()
    val_proba = model.predict_proba(X_all[val_mask])[:, 1]
    val_tile_ids = sorted(val_df["TILE_ID"].unique())

    print(f"\n[INFO] Sweeping decision threshold on {len(val_tile_ids)} val tiles "
          f"(maximizing pixel-level IoU) ...")
    val_cache = _build_tile_cache(val_tile_ids, tiff_dir, labels_1d_dir, classical_results_dir)
    sweep_df = _sweep_threshold(val_df, val_proba, val_tile_ids, val_cache)
    sweep_df.to_csv(results_dir / "phase1_xgb_threshold_sweep_val.csv", index=False)
    print(sweep_df.to_string(index=False))

    best_row = sweep_df.loc[sweep_df["iou_slick"].idxmax()]
    best_threshold = float(best_row["threshold"])
    print(f"\n[INFO] Best threshold on val: {best_threshold} "
          f"(IoU={best_row['iou_slick']:.3f}, precision={best_row['precision']:.3f}, "
          f"recall={best_row['recall']:.3f})")

    # ── Pixel-level evaluation on the test split (comparable to BASELINE_ML) ─
    test_df = full_df[test_mask].copy()
    test_proba = model.predict_proba(X_all[test_mask])[:, 1]
    test_tile_ids = sorted(test_df["TILE_ID"].unique())
    test_cache = _build_tile_cache(test_tile_ids, tiff_dir, labels_1d_dir, classical_results_dir)

    keep_idx_raw = {
        tile_id: set(g["POLY_IDX"]) for tile_id, g in test_df.groupby("TILE_ID")
    }
    keep_idx_default = _keep_idx_at_threshold(test_df, test_proba, 0.5)
    keep_idx_tuned = _keep_idx_at_threshold(test_df, test_proba, best_threshold)

    print(f"\n[INFO] Pixel-level evaluation on {len(test_tile_ids)} test tiles ...")
    metrics_raw = _pixel_metrics_from_cache(test_tile_ids, keep_idx_raw, test_cache)
    metrics_default = _pixel_metrics_from_cache(test_tile_ids, keep_idx_default, test_cache)
    metrics_tuned = _pixel_metrics_from_cache(test_tile_ids, keep_idx_tuned, test_cache)

    pixel_df = pd.DataFrame([
        {"variant": "felzenszwalb_raw_test_tiles", "threshold": None, **metrics_raw},
        {"variant": "felzenszwalb_plus_xgb_default_0.5", "threshold": 0.5, **metrics_default},
        {"variant": "felzenszwalb_plus_xgb_tuned_on_val", "threshold": best_threshold, **metrics_tuned},
    ])
    pixel_df.to_csv(results_dir / "phase1_xgb_test_pixel_metrics.csv", index=False)

    print(f"\n[DONE] Results -> {results_dir}")
    print(pixel_df.to_string(index=False))

    # ── Side-by-side vs. the registered Random Forest result ────────────────
    rf_metrics_path = results_dir / "phase1_test_pixel_metrics.csv"
    if rf_metrics_path.exists():
        rf_df = pd.read_csv(rf_metrics_path)
        rf_tuned = rf_df[rf_df["variant"] == "felzenszwalb_plus_rf_tuned_on_val"].iloc[0]
        xgb_tuned = pixel_df[pixel_df["variant"] == "felzenszwalb_plus_xgb_tuned_on_val"].iloc[0]
        print("\n[COMPARISON] Random Forest (registered) vs. XGBoost (this experiment), "
              "both tuned on val, test split:")
        comp = pd.DataFrame([
            {"model": "random_forest", "threshold": rf_tuned["threshold"],
             "iou_slick": rf_tuned["iou_slick"], "precision": rf_tuned["precision"],
             "recall": rf_tuned["recall"]},
            {"model": "xgboost", "threshold": xgb_tuned["threshold"],
             "iou_slick": xgb_tuned["iou_slick"], "precision": xgb_tuned["precision"],
             "recall": xgb_tuned["recall"]},
        ])
        comp.to_csv(results_dir / "phase1_rf_vs_xgb_comparison.csv", index=False)
        print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
