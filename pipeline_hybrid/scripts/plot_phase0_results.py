"""
Generates figures summarizing the Phase 0 feasibility check (see
phase0_fg_bg_feasibility.py and ../README.md). Reads the CSVs that script
already wrote to results/, plus the source dataset CSV for the raw FG/BG
distribution plot, and re-runs nothing.

Usage (from pipeline_hybrid/, after phase0_fg_bg_feasibility.py has run):
  python scripts/plot_phase0_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPTS_DIR.parent / "config.json"

# Validated 2-color categorical pair (dataviz skill default palette, slots 1/2).
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


def plot_feature_importance(results_dir: Path, out_dir: Path) -> None:
    imp = pd.read_csv(results_dir / "rf_feature_importances.csv", index_col=0)
    imp = imp.sort_values("importance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp["importance"], color=_BLUE, height=0.6)
    for y, v in enumerate(imp["importance"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (Random Forest, full-data fit)")
    ax.set_title("Which FG/BG radiometric fields separate oil from sea?")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_feature_importance.png", dpi=200)
    plt.close(fig)


def plot_model_comparison(results_dir: Path, out_dir: Path) -> None:
    baseline = pd.read_csv(results_dir / "baseline_mean_only_metrics.csv")
    rf = pd.read_csv(results_dir / "rf_cv_metrics.csv")
    baseline_mean = baseline[baseline["fold"] == "mean"].iloc[0]
    rf_mean = rf[rf["fold"] == "mean"].iloc[0]

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_a = ax.bar(x - width / 2, [baseline_mean[m] for m in metrics], width,
                     label="Baseline: MEAN only (LogReg)", color=_ORANGE)
    bars_b = ax.bar(x + width / 2, [rf_mean[m] for m in metrics], width,
                     label="Random Forest: 7 fields", color=_BLUE)
    for bars in (bars_a, bars_b):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                     f"{b.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (5-fold GroupKFold mean, by scene)")
    ax.set_title("Phase 0: naive intensity threshold vs. Random Forest")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_model_comparison.png", dpi=200)
    plt.close(fig)


def plot_fg_bg_distributions(dataset_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(dataset_csv, sep=";")
    fields = ["MAX", "MEAN"]  # top-2 by importance

    fig, axes = plt.subplots(1, len(fields), figsize=(11, 4))
    for ax, field in zip(axes, fields):
        fg = df[f"FG_{field}"]
        bg = df[f"BG_{field}"]
        bins = np.linspace(min(fg.min(), bg.min()), max(fg.max(), bg.max()), 30)
        ax.hist(bg, bins=bins, alpha=0.65, color=_BLUE, label="BG (sea)")
        ax.hist(fg, bins=bins, alpha=0.65, color=_ORANGE, label="FG (oil)")
        ax.set_xlabel(field)
        ax.set_ylabel("Polygons (n=385)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("FG (oil) vs. BG (sea) pixel-intensity distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_fg_bg_distributions.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(results_dir: Path, out_dir: Path) -> None:
    cm = pd.read_csv(results_dir / "rf_confusion_matrix.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm.values, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Random Forest — aggregated over 5 folds")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm.values[i, j]
            color = "white" if val > cm.values.max() / 2 else _TEXT
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=13)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_confusion_matrix.png", dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = _load_config()
    config_dir = _CONFIG_PATH.parent
    dataset_root = (config_dir / cfg["dataset_root"]).resolve()
    dataset_csv = dataset_root / cfg["csv_path"]
    results_dir = (config_dir / cfg["results_dir"]).resolve()

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_feature_importance(results_dir, out_dir)
    plot_model_comparison(results_dir, out_dir)
    plot_fg_bg_distributions(dataset_csv, out_dir)
    plot_confusion_matrix(results_dir, out_dir)

    print(f"[DONE] Figures -> {out_dir}")


if __name__ == "__main__":
    main()
