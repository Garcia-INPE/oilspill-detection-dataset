"""
Plots results/dark_percentile_sweep.csv (see sweep_dark_percentile.py):
IoU/precision/recall vs. dark_percentile, marking the current config.json
default (30) and the empirical best found in the sweep.

Usage (from pipeline_classical/):
  python scripts/plot_dark_percentile_sweep.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _SCRIPTS_DIR.parent / "results"
_PREVIOUS_DEFAULT = 30

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GREEN = "#4e9c6d"
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


def main() -> None:
    df = pd.read_csv(_RESULTS_DIR / "dark_percentile_sweep.csv")
    out_dir = _RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_row = df.loc[df["iou_slick"].idxmax()]
    best_pct = int(best_row["dark_percentile"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["dark_percentile"], df["iou_slick"], color=_BLUE, linewidth=2.5,
            marker="o", markersize=5, label="IoU (primary metric)")
    ax.plot(df["dark_percentile"], df["precision"], color=_ORANGE, linewidth=1.5,
            marker="o", markersize=4, label="Precision")
    ax.plot(df["dark_percentile"], df["recall"], color=_GREEN, linewidth=1.5,
            marker="o", markersize=4, label="Recall")

    ax.axvline(_PREVIOUS_DEFAULT, color=_GRID, linestyle="--", linewidth=1.5)
    ax.text(_PREVIOUS_DEFAULT + 1, 0.95, f"previous default ({_PREVIOUS_DEFAULT})",
            fontsize=9, color=_TEXT, va="top")

    if best_pct != _PREVIOUS_DEFAULT:
        ax.axvline(best_pct, color=_BLUE, linestyle=":", linewidth=1.5, alpha=0.6)
        ax.text(best_pct + 1, 0.88, f"adopted, best IoU ({best_pct})",
                fontsize=9, color=_BLUE, va="top")

    ax.set_xlabel("dark_percentile")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Felzenszwalb dark_percentile sweep\n(pixel-level, all 173 tiles, scale/sigma/min_size fixed)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path = out_dir / "dark_percentile_sweep.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[DONE] Figure -> {out_path}")


if __name__ == "__main__":
    main()
