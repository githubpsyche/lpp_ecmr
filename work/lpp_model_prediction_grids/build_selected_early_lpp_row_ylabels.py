"""Render the selected Early-LPP bars with one y label per panel row."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_selected_early_lpp_figure import (
    CATEGORIES,
    COLORS,
    EDGE_COLOR,
    ERROR_COLOR,
    GRID_COLOR,
    POSITIONS,
    SOURCES,
    STYLE,
    add_bracket,
)


WORK_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = WORK_DIR / "original_early_lpp_summary.csv"
OUTPUT_STEM = WORK_DIR / "selected_early_lpp_row_ylabels"


def load_rows() -> list[dict[str, str]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}; run build_selected_early_lpp_figure.py first"
        )
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {
        (source.source_id, category, status)
        for source in SOURCES
        for category, _ in CATEGORIES
        for status in ("Remembered", "Forgotten")
    }
    selected = [
        row
        for row in rows
        if (
            row["source_id"],
            row["category"],
            row["recall_status"],
        )
        in expected
        and row["metric"] == "original_z_early_lpp"
    ]
    actual = {
        (row["source_id"], row["category"], row["recall_status"])
        for row in selected
    }
    if actual != expected or len(selected) != 16:
        raise ValueError(
            f"Expected 16 selected Early-LPP cells; missing={expected - actual}, "
            f"extra={actual - expected}"
        )
    for row in selected:
        mean, lower, upper = map(
            float,
            (row["mean"], row["lower"], row["upper"]),
        )
        if not np.isfinite([mean, lower, upper]).all() or not lower <= mean <= upper:
            raise ValueError(f"Invalid summary row: {row}")
    return selected


def plot(rows: list[dict[str, str]]) -> None:
    lookup = {
        (row["source_id"], row["category"], row["recall_status"]): row
        for row in rows
    }
    value_min = min(float(row["lower"]) for row in rows)
    value_max = max(float(row["upper"]) for row in rows)
    data_range = value_max - min(0.0, value_min)
    bracket_gap = 0.075 * data_range
    bracket_tick = 0.030 * data_range
    label_allowance = 0.10 * data_range
    y_min = min(-0.05, value_min - 0.06 * data_range)
    y_max = value_max + 2 * bracket_gap + label_allowance

    with plt.rc_context(STYLE):
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(8.8, 7.0),
            sharey=True,
        )
        axes = axes_grid.ravel()
        for panel_index, (axis, source) in enumerate(
            zip(axes, SOURCES, strict=True)
        ):
            source_rows = [
                lookup[(source.source_id, category, status)]
                for category, _ in CATEGORIES
                for status in ("Remembered", "Forgotten")
            ]
            means = np.asarray([float(row["mean"]) for row in source_rows])
            lower = np.asarray([float(row["lower"]) for row in source_rows])
            upper = np.asarray([float(row["upper"]) for row in source_rows])
            errors = np.vstack((means - lower, upper - means))

            axis.bar(
                POSITIONS,
                means,
                width=0.56,
                color=COLORS,
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                yerr=errors,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": 1.05,
                    "capsize": 2.7,
                },
                zorder=3,
            )
            axis.set_title(source.title, fontsize=10.5, pad=10, linespacing=1.18)
            axis.set_xticks(
                POSITIONS,
                [
                    "Negative\nRemembered",
                    "Negative\nForgotten",
                    "Neutral\nRemembered",
                    "Neutral\nForgotten",
                ],
            )
            axis.tick_params(axis="x", length=0, pad=4, labelsize=8.4)
            axis.set_xlim(-0.52, 3.58)
            axis.set_ylim(y_min, y_max)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            axis.axhline(0, color="#8C8C8C", linewidth=0.8, zorder=1)
            if panel_index % 2 == 1:
                axis.tick_params(axis="y", labelleft=False)
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

            for start in (0, 2):
                difference = float(means[start] - means[start + 1])
                expected_difference = float(source_rows[start]["mean"]) - float(
                    source_rows[start + 1]["mean"]
                )
                if not np.isclose(
                    difference,
                    expected_difference,
                    rtol=0,
                    atol=1e-12,
                ):
                    raise AssertionError(f"Delta mismatch for {source.source_id}")
                add_bracket(
                    axis,
                    POSITIONS[start],
                    POSITIONS[start + 1],
                    float(max(upper[start], upper[start + 1]) + bracket_gap),
                    bracket_tick,
                    difference,
                )

        for axis in (axes[0], axes[2]):
            axis.set_ylabel("Early LPP amplitude (z)", labelpad=8)
        figure.subplots_adjust(
            left=0.105,
            right=0.99,
            top=0.93,
            bottom=0.09,
            hspace=0.42,
            wspace=0.18,
        )
        figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    rows = load_rows()
    plot(rows)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
