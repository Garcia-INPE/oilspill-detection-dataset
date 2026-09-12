"""
Phase 1: label Felzenszwalb's own detected candidates (not the manually
digitized Oil_Stats polygons used in Phase 0) by spatial IoU against
ground truth, train a classifier to reject false-positive candidates, and
evaluate with the same pixel-level protocol as the other two baselines.

Labeling: a candidate polygon is positive (real oil) if its IoU against the
best-matching ground-truth polygon in the same tile is >= min_iou_positive
(config.json, default 0.3). This is stricter than "any overlap" and
penalizes both under- and over-segmented candidates.

Candidate features are reused as-is from pipeline_classical/scripts/
run_batch.py's prior run (BASELINE_CLASSICAL, 9242 candidates across 173
tiles, same felzenszwalb config as pipeline_classical/config.json) — no
resegmentation here.

Train/val/test tiles follow the dataset's own splits/*.csv (the same split
pipeline_ml's BASELINE_ML was evaluated on), so the final pixel-level
metric on the test tiles is directly comparable to BASELINE_ML/
BASELINE_CLASSICAL in pipeline_comparison.csv once/if this pipeline is
promoted to a registered benchmark.

Usage (from pipeline_hybrid/):
  python scripts/phase1_candidate_classifier.py
"""
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"
_EPS = 1e-9

_ID_COLS = ["TILE_ID", "ALGORITHM", "POLY_IDX"]
_DROP_COLS = ["CENTR_KM_LAT", "CENTR_KM_LON", "UTM_ZONE", "AREA_KM_DS", "PERIM_KM_DS"]
_CATEGORICAL_COLS = ["FG_BG_KS_RES", "FG_BG_MW_RES"]


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate labeling (IoU vs. best-matching ground-truth polygon)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(geom_a, geom_b) -> float:
    if geom_a is None or geom_b is None or geom_a.is_empty or geom_b.is_empty:
        return 0.0
    if not geom_a.intersects(geom_b):
        return 0.0
    inter = geom_a.intersection(geom_b).area
    union = geom_a.union(geom_b).area
    return inter / union if union > 0 else 0.0


def _label_tile_candidates(candidate_gdf: gpd.GeoDataFrame, gt_geojson_path: Path,
                            min_iou: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(candidate_gdf)
    if not gt_geojson_path.exists():
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    gt_gdf = gpd.read_file(gt_geojson_path)
    gt_geoms = [g for g in gt_gdf.geometry if g is not None and not g.is_empty]
    if not gt_geoms:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    max_ious = np.zeros(n, dtype=float)
    for i, cand_geom in enumerate(candidate_gdf.geometry):
        max_ious[i] = max(_iou(cand_geom, gt_geom) for gt_geom in gt_geoms)

    labels = (max_ious >= min_iou).astype(int)
    return labels, max_ious


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def _load_split_map(splits_dir: Path) -> dict:
    split_map = {}
    for split_name in ("train", "val", "test"):
        split_df = pd.read_csv(splits_dir / f"{split_name}.csv")
        for sample_id in split_df["sample_id"]:
            split_map[sample_id] = split_name
    return split_map


def _build_dataset(cfg: dict, config_dir: Path) -> pd.DataFrame:
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    classical_results_dir = (config_dir / cfg["classical_results_dir"]).resolve()
    gt_geojson_dir = dataset_root / cfg["labels_vector_dir"]
    splits_dir = dataset_root / cfg["splits_dir"]
    min_iou = float(cfg.get("min_iou_positive", 0.3))

    split_map = _load_split_map(splits_dir)

    feature_files = sorted(
        (classical_results_dir / "BASELINE_CLASSICAL").glob("*/*_features.csv"))
    if not feature_files:
        print(f"[ERROR] No candidate feature files found under {classical_results_dir}")
        print("        Run pipeline_classical/scripts/run_batch.py first.")
        sys.exit(1)

    rows = []
    for feat_path in tqdm(feature_files, desc="Labeling tiles", unit="tile"):
        tile_id = feat_path.stem.replace("_features", "")
        feat_df = pd.read_csv(feat_path, sep=";")

        geojson_path = feat_path.parent / f"{tile_id}.geojson"
        candidate_gdf = gpd.read_file(geojson_path)
        if len(candidate_gdf) != len(feat_df):
            print(f"[WARN] {tile_id}: candidate count mismatch "
                  f"(geojson={len(candidate_gdf)}, features={len(feat_df)}) — skipping tile")
            continue

        labels, max_ious = _label_tile_candidates(
            candidate_gdf, gt_geojson_dir / f"{tile_id}.geojson", min_iou)

        feat_df["label"] = labels
        feat_df["max_iou"] = max_ious
        scene_match = re.match(r"^(IMG_\d+)", tile_id)
        feat_df["scene"] = scene_match.group(1) if scene_match else tile_id
        feat_df["split"] = split_map.get(tile_id, "unknown")
        rows.append(feat_df)

    full_df = pd.concat(rows, ignore_index=True)

    n_unknown = (full_df["split"] == "unknown").sum()
    if n_unknown:
        print(f"[WARN] {n_unknown} candidates belong to tiles absent from splits/*.csv "
              f"— excluded from train/val/test use")

    return full_df


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    numeric_cols = [
        c for c in df.columns
        if c not in _ID_COLS + _DROP_COLS + _CATEGORICAL_COLS
        + ["label", "max_iou", "scene", "split"]
    ]
    X_numeric = df[numeric_cols].to_numpy(dtype=float)

    cat_blocks = []
    cat_names = []
    for col in _CATEGORICAL_COLS:
        cat_blocks.append((df[col] == "DIFF").to_numpy(dtype=float).reshape(-1, 1))
        cat_names.append(f"{col}_DIFF")

    X = np.hstack([X_numeric] + cat_blocks)
    feature_names = numeric_cols + cat_names
    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated feasibility check (train split only, GroupKFold by scene)
# ─────────────────────────────────────────────────────────────────────────────

def _cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     n_splits: int, random_state: int) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state)
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


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level evaluation (mirrors pipeline_classical/scripts/compare_to_ground_truth.py)
# ─────────────────────────────────────────────────────────────────────────────

def _ground_truth_mask(labels_1d_dir: Path, tile_id: str) -> np.ndarray:
    arr = np.array(Image.open(labels_1d_dir / f"{tile_id}_1D.png"))
    return arr > 0


def _build_tile_cache(tile_ids: list, tiff_dir: Path, labels_1d_dir: Path,
                       classical_results_dir: Path) -> dict:
    """Preload per-tile geometry/raster artifacts once, reused across every
    threshold tried in the sweep (avoids re-reading files per threshold)."""
    cache = {}
    for tile_id in tile_ids:
        with rasterio.open(tiff_dir / f"{tile_id}.tiff") as src:
            transform, shape = src.transform, src.shape
        gt = _ground_truth_mask(labels_1d_dir, tile_id)
        geojson_path = classical_results_dir / "BASELINE_CLASSICAL" / tile_id / f"{tile_id}.geojson"
        gdf = gpd.read_file(geojson_path)
        cache[tile_id] = {"gdf": gdf, "transform": transform, "shape": shape, "gt": gt}
    return cache


def _predicted_mask_from_cache(entry: dict, keep_poly_idx: set) -> np.ndarray:
    if not keep_poly_idx:
        return np.zeros(entry["shape"], dtype=bool)
    geoms = [entry["gdf"].geometry.iloc[i] for i in sorted(keep_poly_idx) if i < len(entry["gdf"])]
    if not geoms:
        return np.zeros(entry["shape"], dtype=bool)
    mask = rasterize([(g, 1) for g in geoms], out_shape=entry["shape"],
                      transform=entry["transform"], fill=0, dtype="uint8")
    return mask > 0


def _pixel_metrics_from_cache(tile_ids: list, keep_idx_by_tile: dict, cache: dict) -> dict:
    tp = fp = fn = tn = 0
    for tile_id in tile_ids:
        entry = cache[tile_id]
        pred = _predicted_mask_from_cache(entry, keep_idx_by_tile.get(tile_id, set()))
        gt = entry["gt"]

        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
        tn += int(np.logical_and(~pred, ~gt).sum())

    return {
        "iou_slick": tp / (tp + fp + fn + _EPS),
        "precision": tp / (tp + fp + _EPS),
        "recall": tp / (tp + fn + _EPS),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + _EPS),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _keep_idx_at_threshold(df: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict:
    keep = df.loc[proba >= threshold].groupby("TILE_ID")["POLY_IDX"].apply(set)
    return keep.to_dict()


def _sweep_threshold(df: pd.DataFrame, proba: np.ndarray, tile_ids: list,
                      cache: dict) -> pd.DataFrame:
    thresholds = np.concatenate([
        np.arange(0.001, 0.01, 0.001),
        np.arange(0.01, 0.05, 0.01),
        np.arange(0.05, 1.00, 0.05),
    ])
    rows = []
    for threshold in np.round(thresholds, 3):
        keep_idx = _keep_idx_at_threshold(df, proba, threshold)
        metrics = _pixel_metrics_from_cache(tile_ids, keep_idx, cache)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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
    min_iou = float(cfg.get("min_iou_positive", 0.3))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Labeling candidates (min_iou_positive={min_iou}) ...")
    full_df = _build_dataset(cfg, config_dir)
    full_df = full_df[full_df["split"] != "unknown"].reset_index(drop=True)

    print(f"[INFO] Total candidates: {len(full_df)}")
    print(full_df.groupby("split")["label"].agg(["count", "sum", "mean"])
          .rename(columns={"count": "n", "sum": "n_positive", "mean": "positive_rate"}))

    full_df.to_csv(results_dir / "phase1_candidates_labeled.csv", index=False)

    X_all, feature_names = _feature_matrix(full_df)
    y_all = full_df["label"].to_numpy()

    train_mask = (full_df["split"] == "train").to_numpy()
    val_mask = (full_df["split"] == "val").to_numpy()
    test_mask = (full_df["split"] == "test").to_numpy()

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    groups_train = full_df.loc[train_mask, "scene"].to_numpy()

    # ── Cross-validated feasibility check on the train split ────────────────
    print("[INFO] Running GroupKFold cross-validation on the train split ...")
    cv_df = _cross_validate(X_train, y_train, groups_train, n_splits, random_state)
    cv_df.to_csv(results_dir / "phase1_cv_metrics.csv", index=False)
    print(cv_df.to_string(index=False))

    # ── Final model: fit on full train split ─────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)

    importances = pd.DataFrame({
        "feature": feature_names, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(results_dir / "phase1_feature_importances.csv", index=False)

    # ── Candidate-level classification report on val/test ───────────────────
    report_rows = []
    for split_name, mask in (("val", val_mask), ("test", test_mask)):
        if mask.sum() == 0:
            continue
        y_true = y_all[mask]
        y_pred = model.predict(X_all[mask])
        y_proba = model.predict_proba(X_all[mask])[:, 1]
        report_rows.append({
            "split": split_name,
            "n_candidates": int(mask.sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        })
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(results_dir / "phase1_val_test_classification_report.csv", index=False)
    print("\n[INFO] Candidate-level classification report (val/test):")
    print(report_df.to_string(index=False))

    cm = confusion_matrix(y_all[test_mask], model.predict(X_all[test_mask]))
    pd.DataFrame(cm, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"]) \
        .to_csv(results_dir / "phase1_test_confusion_matrix.csv")

    # ── Threshold calibration on the val split (maximize pixel-level IoU) ────
    val_df = full_df[val_mask].copy()
    val_proba = model.predict_proba(X_all[val_mask])[:, 1]
    val_tile_ids = sorted(val_df["TILE_ID"].unique())

    print(f"\n[INFO] Sweeping decision threshold on {len(val_tile_ids)} val tiles "
          f"(maximizing pixel-level IoU) ...")
    val_cache = _build_tile_cache(val_tile_ids, tiff_dir, labels_1d_dir, classical_results_dir)
    sweep_df = _sweep_threshold(val_df, val_proba, val_tile_ids, val_cache)
    sweep_df.to_csv(results_dir / "phase1_threshold_sweep_val.csv", index=False)
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
    keep_idx_rf_default = _keep_idx_at_threshold(test_df, test_proba, 0.5)
    keep_idx_rf_tuned = _keep_idx_at_threshold(test_df, test_proba, best_threshold)

    test_df["rf_proba"] = test_proba
    test_df["kept_tuned"] = 0
    for tile_id, idx_set in keep_idx_rf_tuned.items():
        rows_mask = (test_df["TILE_ID"] == tile_id) & (test_df["POLY_IDX"].isin(idx_set))
        test_df.loc[rows_mask, "kept_tuned"] = 1
    test_df[["TILE_ID", "POLY_IDX", "label", "max_iou", "rf_proba", "kept_tuned"]].to_csv(
        results_dir / "phase1_test_candidate_predictions.csv", index=False)

    print(f"\n[INFO] Pixel-level evaluation on {len(test_tile_ids)} test tiles ...")
    metrics_raw = _pixel_metrics_from_cache(test_tile_ids, keep_idx_raw, test_cache)
    metrics_rf_default = _pixel_metrics_from_cache(test_tile_ids, keep_idx_rf_default, test_cache)
    metrics_rf_tuned = _pixel_metrics_from_cache(test_tile_ids, keep_idx_rf_tuned, test_cache)

    pixel_df = pd.DataFrame([
        {"variant": "felzenszwalb_raw_test_tiles", "threshold": None, **metrics_raw},
        {"variant": "felzenszwalb_plus_rf_default_0.5", "threshold": 0.5, **metrics_rf_default},
        {"variant": "felzenszwalb_plus_rf_tuned_on_val", "threshold": best_threshold, **metrics_rf_tuned},
    ])
    pixel_df.to_csv(results_dir / "phase1_test_pixel_metrics.csv", index=False)

    print(f"\n[DONE] Results -> {results_dir}")
    print(pixel_df.to_string(index=False))


if __name__ == "__main__":
    main()
