"""Render the Early-LPP candidate using the shared four-panel model set."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lpp_ecmr.data_contract import MIXED_EXPECTED_SUBJECTS

from build_original_early_lpp_summaries import (
    CATEGORIES,
    MEMORY_STATUSES,
)


WORK_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = WORK_DIR / "original_early_lpp_summary.csv"
CONTRAST_PATH = WORK_DIR / "original_early_lpp_contrasts.csv"
FILTERED_SUMMARY = WORK_DIR / "matched_models_early_lpp_summary.csv"
OUTPUT_STEM = WORK_DIR / "matched_models_early_lpp_figure"

CORAL = "#ED706B"
PALE_PINK = "#F4B5B4"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#B3B3B3"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
COLORS = (CORAL, PALE_PINK, DARK_GRAY, LIGHT_GRAY)
POSITIONS = np.asarray([0.0, 0.88, 2.18, 3.06])
STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


@dataclass(frozen=True)
class FigureSource:
    source_id: str
    source_type: str
    title: str


SOURCES = (
    FigureSource("Observed", "observed", "Observed data"),
    FigureSource(
        "CategoryOnly_eCMR_LPP_EmotionalOnly",
        "predicted",
        "LPP-based learning boost\nfor negative items only",
    ),
    FigureSource(
        "EEM_eCMR",
        "predicted",
        "Emotion-based\nlearning boost",
    ),
    FigureSource(
        "EEM_eCMR_LPP_EmotionalOnly",
        "predicted",
        "Emotion-based + LPP-based\nlearning boost for negative items only",
    ),
)


def _format_delta(value: float) -> str:
    rounded = round(value, 2)
    if rounded == 0 and not np.isclose(value, 0.0, rtol=0, atol=5e-6):
        return r"$\Delta\approx0$"
    return rf"$\Delta={value:.2f}$"


def add_bracket(
    axis: plt.Axes,
    left: float,
    right: float,
    height: float,
    tick: float,
    difference: float,
) -> float:
    axis.plot(
        [left, left, right, right],
        [height - tick, height, height, height - tick],
        color=EDGE_COLOR,
        linewidth=0.9,
        clip_on=False,
    )
    label_height = height + tick * 0.45
    axis.text(
        (left + right) / 2,
        label_height,
        _format_delta(difference),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=EDGE_COLOR,
    )
    return label_height


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def load_rows() -> tuple[list[str], list[dict[str, str]]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}; run "
            "build_original_early_lpp_summaries.py first"
        )
    fieldnames, rows = _read_csv(SUMMARY_PATH)
    source_by_id = {source.source_id: source for source in SOURCES}
    expected = {
        (source.source_id, category, status)
        for source in SOURCES
        for category, _code in CATEGORIES
        for status, _remembered in MEMORY_STATUSES
    }
    selected = [
        row
        for row in rows
        if row["metric"] == "original_z_early_lpp"
        and (row["source_id"], row["category"], row["recall_status"])
        in expected
    ]
    actual = [
        (row["source_id"], row["category"], row["recall_status"])
        for row in selected
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate matched-model Early-LPP rows")
    if set(actual) != expected or len(selected) != 16:
        raise ValueError(
            f"Expected 16 matched-model Early-LPP cells; "
            f"missing={expected - set(actual)}, extra={set(actual) - expected}"
        )

    for row in selected:
        source = source_by_id[row["source_id"]]
        if row["source_type"] != source.source_type:
            raise ValueError(f"Unexpected source type: {row}")
        mean, lower, upper = map(
            float,
            (row["mean"], row["lower"], row["upper"]),
        )
        if not np.isfinite([mean, lower, upper]).all() or not lower <= mean <= upper:
            raise ValueError(f"Invalid mean or interval: {row}")
        if float(row["confidence_level"]) != 0.95:
            raise ValueError(f"Unexpected interval level: {row}")
        if source.source_type == "observed":
            if row["interval"] != "Percentile bootstrap across participant cell means":
                raise ValueError(f"Unexpected observed interval: {row}")
            if int(row["n_units"]) != MIXED_EXPECTED_SUBJECTS:
                raise ValueError(f"Unexpected observed sample size: {row}")
        else:
            if row["interval"] != (
                "Central interval across complete simulated datasets"
            ):
                raise ValueError(f"Unexpected prediction interval: {row}")
            if int(row["n_units"]) != 200:
                raise ValueError(f"Unexpected prediction count: {row}")

    source_order = {source.source_id: index for index, source in enumerate(SOURCES)}
    category_order = {
        category: index for index, (category, _code) in enumerate(CATEGORIES)
    }
    memory_order = {
        status: index for index, (status, _remembered) in enumerate(MEMORY_STATUSES)
    }
    selected.sort(
        key=lambda row: (
            source_order[row["source_id"]],
            category_order[row["category"]],
            memory_order[row["recall_status"]],
        )
    )
    return fieldnames, selected


def load_contrasts() -> dict[tuple[str, str], float]:
    _fieldnames, rows = _read_csv(CONTRAST_PATH)
    source_ids = {source.source_id for source in SOURCES}
    expected = {
        (source_id, category)
        for source_id in source_ids
        for category, _code in CATEGORIES
    }
    contrasts: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["source_id"] not in source_ids:
            continue
        prefix = "Remembered minus Forgotten Early LPP: "
        if not row["contrast"].startswith(prefix):
            continue
        category = row["contrast"][len(prefix) :]
        key = (row["source_id"], category)
        if key in contrasts:
            raise ValueError(f"Duplicate Early-LPP contrast: {key}")
        contrasts[key] = float(row["estimate"])
    if set(contrasts) != expected:
        raise ValueError(
            f"Matched-model Early-LPP contrast mismatch; "
            f"missing={expected - set(contrasts)}, extra={set(contrasts) - expected}"
        )
    return contrasts


def write_summary(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with FILTERED_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, str]],
    contrasts: dict[tuple[str, str], float],
) -> None:
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
                for category, _code in CATEGORIES
                for status, _remembered in MEMORY_STATUSES
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

            for start, (category, _code) in zip((0, 2), CATEGORIES, strict=True):
                difference = float(means[start] - means[start + 1])
                expected_difference = contrasts[(source.source_id, category)]
                if not np.isclose(
                    difference,
                    expected_difference,
                    rtol=0,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        f"Contrast mismatch for {source.source_id}, {category}: "
                        f"plotted={difference}, expected={expected_difference}"
                    )
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
    fieldnames, rows = load_rows()
    contrasts = load_contrasts()
    write_summary(fieldnames, rows)
    plot(rows, contrasts)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved {FILTERED_SUMMARY}")


if __name__ == "__main__":
    main()
