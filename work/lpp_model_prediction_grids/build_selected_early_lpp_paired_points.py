"""Plot a paired-point candidate for the selected Early-LPP comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


WORK_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = WORK_DIR / "original_early_lpp_summary.csv"
OUTPUT_STEM = WORK_DIR / "selected_early_lpp_paired_points"

CORAL = "#ED706B"
PALE_PINK = "#F4B5B4"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#B3B3B3"
EDGE_COLOR = "#333333"
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
CATEGORIES = (
    ("Negative", 1.0, CORAL, PALE_PINK),
    ("Neutral", 0.0, DARK_GRAY, LIGHT_GRAY),
)
STATUS_OFFSET = {"Remembered": 0.075, "Forgotten": -0.075}
STATUS_MARKER = {"Remembered": "o", "Forgotten": "s"}

STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


def load_rows() -> dict[tuple[str, str, str], dict[str, float]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}; run build_selected_early_lpp_figure.py first"
        )
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected_ids = {source_id for source_id, _ in SOURCES}
    selected = [
        row
        for row in rows
        if row["source_id"] in selected_ids
        and row["metric"] == "original_z_early_lpp"
    ]
    expected = {
        (source_id, category, status)
        for source_id, _ in SOURCES
        for category, *_ in CATEGORIES
        for status in STATUS_OFFSET
    }
    actual = {
        (row["source_id"], row["category"], row["recall_status"])
        for row in selected
    }
    if actual != expected or len(selected) != 16:
        raise ValueError(
            f"Expected 16 selected Early-LPP cells; missing={expected - actual}, "
            f"extra={actual - expected}"
        )

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in selected:
        key = (row["source_id"], row["category"], row["recall_status"])
        values = {
            name: float(row[name]) for name in ("mean", "lower", "upper")
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Non-finite summary row: {row}")
        if not values["lower"] <= values["mean"] <= values["upper"]:
            raise ValueError(f"Invalid interval: {row}")
        lookup[key] = values
    return lookup


def plot(lookup: dict[tuple[str, str, str], dict[str, float]]) -> None:
    all_intervals = [
        value
        for row in lookup.values()
        for value in (row["lower"], row["upper"])
    ]
    x_min = min(all_intervals)
    x_max = max(all_intervals)
    x_range = x_max - x_min
    x_limits = (x_min - 0.08 * x_range, x_max + 0.20 * x_range)
    delta_x = x_limits[1] - 0.02 * (x_limits[1] - x_limits[0])

    with plt.rc_context(STYLE):
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(8.8, 6.7),
            sharex=True,
            sharey=True,
        )
        axes = axes_grid.ravel()
        for panel_index, (axis, (source_id, title)) in enumerate(
            zip(axes, SOURCES, strict=True)
        ):
            for category, y_center, remembered_color, forgotten_color in CATEGORIES:
                remembered = lookup[(source_id, category, "Remembered")]
                forgotten = lookup[(source_id, category, "Forgotten")]
                remembered_y = y_center + STATUS_OFFSET["Remembered"]
                forgotten_y = y_center + STATUS_OFFSET["Forgotten"]

                axis.plot(
                    [forgotten["mean"], remembered["mean"]],
                    [forgotten_y, remembered_y],
                    color=remembered_color,
                    linewidth=2.2,
                    solid_capstyle="round",
                    zorder=2,
                )
                for status, row, y, color in (
                    ("Forgotten", forgotten, forgotten_y, forgotten_color),
                    ("Remembered", remembered, remembered_y, remembered_color),
                ):
                    error = np.asarray(
                        [
                            [row["mean"] - row["lower"]],
                            [row["upper"] - row["mean"]],
                        ]
                    )
                    axis.errorbar(
                        row["mean"],
                        y,
                        xerr=error,
                        fmt=STATUS_MARKER[status],
                        markersize=7.2,
                        markerfacecolor=color,
                        markeredgecolor=EDGE_COLOR,
                        markeredgewidth=0.8,
                        ecolor=EDGE_COLOR,
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )

                difference = remembered["mean"] - forgotten["mean"]
                axis.text(
                    delta_x,
                    y_center,
                    rf"$\Delta={difference:.3f}$",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color=EDGE_COLOR,
                )

            axis.set_title(title, fontsize=10.5, pad=9, linespacing=1.18)
            axis.set_xlim(*x_limits)
            axis.set_ylim(-0.42, 1.42)
            axis.set_yticks([1.0, 0.0], ["Negative", "Neutral"])
            axis.tick_params(axis="y", length=0, pad=7)
            axis.set_axisbelow(True)
            axis.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
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

        legend_handles = (
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=7,
                markerfacecolor="#777777",
                markeredgecolor=EDGE_COLOR,
                label="Remembered",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="s",
                markersize=7,
                markerfacecolor="#C7C7C7",
                markeredgecolor=EDGE_COLOR,
                label="Forgotten",
            ),
        )
        figure.supxlabel("Early LPP amplitude (z)", y=0.055, fontsize=10.5)
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        figure.subplots_adjust(
            left=0.11,
            right=0.985,
            top=0.92,
            bottom=0.14,
            hspace=0.38,
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
