"""Render the recall-rate candidate using the shared four-panel model set."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lpp_ecmr.data_contract import MIXED_EXPECTED_SUBJECTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = WORK_DIR / "matched_models_recall_rate_figure"
FILTERED_SUMMARY = WORK_DIR / "matched_models_recall_summary.csv"
SUMMARY_SOURCE = PROJECT_ROOT / "work" / "pooled_model_summaries" / "diagnostic_summary.csv"
CONTRAST_SOURCE = (
    PROJECT_ROOT / "work" / "pooled_model_summaries" / "diagnostic_contrasts.csv"
)

CORAL = "#ED706B"
DARK_GRAY = "#666666"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
CATEGORIES = (("Negative", CORAL), ("Neutral", DARK_GRAY))
CONTRAST_NAME = "Negative minus Neutral recall rate"
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


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def add_bracket(
    axis: plt.Axes,
    left: float,
    right: float,
    height: float,
    difference: float,
) -> float:
    tick = 0.007
    axis.plot(
        [left, left, right, right],
        [height - tick, height, height, height - tick],
        color=EDGE_COLOR,
        linewidth=0.9,
        clip_on=False,
    )
    label_height = height + 0.006
    axis.text(
        (left + right) / 2,
        label_height,
        rf"$\Delta={difference:.2f}$",
        ha="center",
        va="bottom",
        fontsize=9,
        color=EDGE_COLOR,
    )
    return label_height


def load_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = _read_csv(SUMMARY_SOURCE)
    source_by_id = {source.source_id: source for source in SOURCES}
    expected = {
        (source.source_id, category)
        for source in SOURCES
        for category, _color in CATEGORIES
    }
    selected = [
        row
        for row in rows
        if row["metric"] == "recall_rate"
        and (row["source_id"], row["category"]) in expected
    ]
    actual = [(row["source_id"], row["category"]) for row in selected]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate matched-model recall rows")
    if set(actual) != expected or len(selected) != 8:
        raise ValueError(
            f"Expected eight matched-model recall cells; actual={actual}"
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
            if row["interval"] != "Participant-cluster percentile bootstrap":
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
        category: index for index, (category, _color) in enumerate(CATEGORIES)
    }
    selected.sort(
        key=lambda row: (
            source_order[row["source_id"]],
            category_order[row["category"]],
        )
    )
    return fieldnames, selected


def load_contrasts() -> dict[str, float]:
    _fieldnames, rows = _read_csv(CONTRAST_SOURCE)
    source_ids = {source.source_id for source in SOURCES}
    contrasts = {
        row["source_id"]: float(row["estimate"])
        for row in rows
        if row["source_id"] in source_ids and row["contrast"] == CONTRAST_NAME
    }
    if set(contrasts) != source_ids:
        raise ValueError("Missing matched-model recall contrasts")
    return contrasts


def write_summary(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with FILTERED_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, str]], contrasts: dict[str, float]) -> None:
    lookup = {(row["source_id"], row["category"]): row for row in rows}
    positions = np.asarray([0.0, 0.92])
    colors = [color for _category, color in CATEGORIES]
    labels = [category for category, _color in CATEGORIES]
    highest_label = 0.0

    with plt.rc_context(STYLE):
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(8.8, 6.4),
            sharey=True,
        )
        axes = axes_grid.ravel()
        for panel_index, (axis, source) in enumerate(
            zip(axes, SOURCES, strict=True)
        ):
            source_rows = [
                lookup[(source.source_id, category)]
                for category, _color in CATEGORIES
            ]
            means = np.asarray([float(row["mean"]) for row in source_rows])
            lower = np.asarray([float(row["lower"]) for row in source_rows])
            upper = np.asarray([float(row["upper"]) for row in source_rows])
            errors = np.vstack((means - lower, upper - means))

            axis.bar(
                positions,
                means,
                width=0.62,
                color=colors,
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                yerr=errors,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": 1.1,
                    "capsize": 3,
                },
                zorder=3,
            )
            axis.set_title(source.title, fontsize=10.5, pad=10, linespacing=1.18)
            axis.set_xticks(positions, labels)
            axis.tick_params(axis="x", length=0, pad=4)
            axis.set_xlim(-0.52, 1.44)
            axis.set_ylim(0, 0.62)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            if panel_index % 2 == 1:
                axis.tick_params(axis="y", labelleft=False)
            axis.text(
                -0.12,
                1.015,
                chr(ord("A") + panel_index),
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                clip_on=False,
            )

            plotted_difference = float(means[0] - means[1])
            expected_difference = contrasts[source.source_id]
            if not np.isclose(
                plotted_difference,
                expected_difference,
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Contrast mismatch for {source.source_id}: "
                    f"plotted={plotted_difference}, expected={expected_difference}"
                )
            highest_label = max(
                highest_label,
                add_bracket(
                    axis,
                    positions[0],
                    positions[1],
                    float(np.max(upper) + 0.022),
                    plotted_difference,
                ),
            )

        if highest_label >= 0.62:
            raise AssertionError("Recall-rate annotation exceeds the shared y limit")
        for axis in (axes[0], axes[2]):
            axis.set_ylabel("Recall rate", labelpad=7)
        figure.subplots_adjust(
            left=0.105,
            right=0.99,
            top=0.93,
            bottom=0.085,
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
