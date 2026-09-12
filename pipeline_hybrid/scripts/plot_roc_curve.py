"""
Plots the actual ROC curve (TPR vs. FPR) for the Phase 0 feasibility check,
complementing fig2_model_comparison.png (which only shows the ROC-AUC value
as one bar among five metrics).

phase0_fg_bg_feasibility.py never saved per-sample predicted probabilities
(only aggregated fold metrics), so this script re-runs the same GroupKFold
cross-validation (same random_state/n_splits from config.json, same model
configs) purely to collect out-of-fold predicted probabilities, then pools
them across all folds into one ROC curve per model — no reprocessing of the
feature CSV logic, that's imported directly from phase0_fg_bg_feasibility.py.

Usage (from pipeline_hybrid/, after phase0_fg_bg_feasibility.py has run):
  python scripts/plot_roc_curve.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"
sys.path.insert(0, str(_SCRIPTS_DIR))
from phase0_fg_bg_feasibility import _STAT_FIELDS, _build_fg_bg_dataset  # noqa: E402

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GRID = "#dedcd6"
_TEXT = "#3a3a38"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _TEXT,
    "text.color": _TEXT,
    "xtick.color": _TEXT,
    "ytick.color": _TEXT,
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.8,
    "font.size": 11,
})


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _pooled_oof_proba(model_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       n_splits: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all = np.empty(0, dtype=int)
    y_proba_all = np.empty(0, dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups):
        model = model_factory(random_state)
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        y_true_all = np.concatenate([y_true_all, y[test_idx]])
        y_proba_all = np.concatenate([y_proba_all, y_proba])

    return y_true_all, y_proba_all


def plot_roc_curve(results_dir: Path, out_dir: Path, X: np.ndarray, y: np.ndarray,
                    groups: np.ndarray, n_splits: int, random_state: int) -> None:
    mean_idx = _STAT_FIELDS.index("MEAN")

    y_true_base, y_proba_base = _pooled_oof_proba(
        lambda rs: LogisticRegression(random_state=rs),
        X[:, [mean_idx]], y, groups, n_splits, random_state)
    fpr_base, tpr_base, _ = roc_curve(y_true_base, y_proba_base)
    auc_base = auc(fpr_base, tpr_base)

    y_true_rf, y_proba_rf = _pooled_oof_proba(
        lambda rs: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs),
        X, y, groups, n_splits, random_state)
    fpr_rf, tpr_rf, _ = roc_curve(y_true_rf, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(fpr_base, tpr_base, color=_ORANGE, linewidth=2,
            label=f"Baseline: MEAN only (AUC = {auc_base:.3f})")
    ax.plot(fpr_rf, tpr_rf, color=_BLUE, linewidth=2,
            label=f"Random Forest: 7 fields (AUC = {auc_rf:.3f})")
    ax.plot([0, 1], [0, 1], color=_GRID, linewidth=1.5, linestyle="--",
            label="Chance (AUC = 0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Phase 0: ROC curve\n(pooled out-of-fold predictions, 5-fold GroupKFold)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_roc_curve.png", dpi=200)
    plt.close(fig)

    print(f"[INFO] Pooled ROC-AUC — baseline: {auc_base:.3f}, RF: {auc_rf:.3f}")


def main() -> None:
    cfg = _load_config()
    config_dir = _CONFIG_PATH.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    csv_path = dataset_root / cfg["csv_path"]
    results_dir = (config_dir / cfg["results_dir"]).resolve()
    random_state = int(cfg.get("random_state", 42))
    n_splits = int(cfg.get("n_splits", 5))

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _build_fg_bg_dataset(csv_path)
    X = data[_STAT_FIELDS].to_numpy()
    y = data["label"].to_numpy()
    groups = data["group"].to_numpy()

    plot_roc_curve(results_dir, out_dir, X, y, groups, n_splits, random_state)
    print(f"[DONE] Figure -> {out_dir / 'fig5_roc_curve.png'}")


if __name__ == "__main__":
    main()
