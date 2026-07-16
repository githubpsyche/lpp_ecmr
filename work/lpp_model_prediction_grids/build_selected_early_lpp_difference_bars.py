"""Plot remembered-minus-forgotten Early-LPP bars for the selected models."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORK_DIR = Path(__file__).resolve().parent
CONTRAST_PATH = WORK_DIR / "original_early_lpp_contrasts.csv"
OUTPUT_STEM = WORK_DIR / "selected_early_lpp_difference_bars"

CORAL = "#ED706B"
DARK_GRAY = "#666666"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"

SOURCES = (
    ("Observed", "Observed data"),
    ("EEM_eCMR", "Emotion-based\nlearning boost"),
    (
        "EEM_eCMR_LPP_General",
        "Emotion-based + LPP-based\nlearning boost for all items",
    ),
    (
        "EEM_eCMR_LPP_EmotionalOnly",
        "Emotion-based + LPP-based\nlearning boost for negative items only",
    ),
)
CATEGORIES = (("Negative", CORAL), ("Neutral", DARK_GRAY))

STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


def load_rows() -> dict[tuple[str, str], dict[str, float]]:
    if not CONTRAST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONTRAST_PATH}; run build_selected_early_lpp_figure.py first"
        )
    with CONTRAST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected_ids = {source_id for source_id, _ in SOURCES}
    expected = {
        (source_id, category)
        for source_id, _ in SOURCES
        for category, _ in CATEGORIES
    }
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id not in selected_ids:
            continue
        prefix = "Remembered minus Forgotten Early LPP: "
        if not row["contrast"].startswith(prefix):
            continue
        category = row["contrast"].removeprefix(prefix)
        key = (source_id, category)
        if key in lookup:
            raise ValueError(f"Duplicate contrast row: {key}")
        values = {
            name: float(row[name]) for name in ("estimate", "lower", "upper")
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Non-finite contrast row: {row}")
        if not values["lower"] <= values["estimate"] <= values["upper"]:
            raise ValueError(f"Invalid paired interval: {row}")
        if float(row["confidence_level"]) != 0.95:
            raise ValueError(f"Unexpected interval level: {row}")
        lookup[key] = values

    if set(lookup) != expected:
        raise ValueError(
            f"Expected eight selected contrast rows; missing={expected - set(lookup)}, "
            f"extra={set(lookup) - expected}"
        )
    return lookup


def plot(lookup: dict[tuple[str, str], dict[str, float]]) -> None:
    rows = [lookup[(source_id, category)] for source_id, _ in SOURCES for category, _ in CATEGORIES]
    lowest = min(float(row["lower"]) for row in rows)
    highest = max(float(row["upper"]) for row in rows)
    value_range = highest - lowest
    y_limits = (min(-0.16, lowest - 0.04 * value_range), highest + 0.10 * value_range)
    positions = np.asarray([0.0, 1.0])

    with plt.rc_context(STYLE):
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(8.5, 6.8),
            sharex=True,
            sharey=True,
        )
        axes = axes_grid.ravel()
        for panel_index, (axis, (source_id, title)) in enumerate(
            zip(axes, SOURCES, strict=True)
        ):
            source_rows = [lookup[(source_id, category)] for category, _ in CATEGORIES]
            estimates = np.asarray([row["estimate"] for row in source_rows])
            lower = np.asarray([row["lower"] for row in source_rows])
            upper = np.asarray([row["upper"] for row in source_rows])
            errors = np.vstack((estimates - lower, upper - estimates))

            axis.bar(
                positions,
                estimates,
                width=0.58,
                color=[color for _, color in CATEGORIES],
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                yerr=errors,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": 1.1,
                    "capsize": 3.0,
                },
                zorder=3,
            )
            for position, estimate in zip(positions, estimates, strict=True):
                label_y = estimate + 0.018 * value_range
                axis.text(
                    position,
                    label_y,
                    f"{estimate:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=EDGE_COLOR,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.4,
                    },
                    zorder=5,
                )

            axis.set_title(title, fontsize=10.5, pad=10, linespacing=1.18)
            axis.set_xticks(positions, [category for category, _ in CATEGORIES])
            axis.tick_params(axis="x", length=0, pad=5)
            axis.set_xlim(-0.55, 1.55)
            axis.set_ylim(*y_limits)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            axis.axhline(0, color="#7F7F7F", linewidth=0.9, zorder=1)
            axis.text(
                -0.12,
                1.03,
                chr(ord("A") + panel_index),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )
            if panel_index % 2 == 1:
                axis.tick_params(axis="y", labelleft=False)

        figure.supylabel(
            "Remembered − forgotten Early LPP (z)",
            x=0.025,
            fontsize=10.5,
        )
        figure.subplots_adjust(
            left=0.11,
            right=0.99,
            top=0.93,
            bottom=0.09,
            hspace=0.40,
            wspace=0.18,
        )
        figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    lookup = load_rows()
    plot(lookup)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
