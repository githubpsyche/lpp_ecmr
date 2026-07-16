"""Plot the selected empirical-versus-predicted recall-rate contrast."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
SUMMARY_SOURCE = PROJECT_ROOT / "work" / "lpp_model_results" / "diagnostic_summary.csv"
CONTRAST_SOURCE = (
    PROJECT_ROOT / "work" / "lpp_model_results" / "diagnostic_contrasts.csv"
)
FILTERED_SUMMARY = WORK_DIR / "selected_recall_summary.csv"
OUTPUT_STEM = WORK_DIR / "selected_recall_rate_figure"

CORAL = "#ED706B"
DARK_GRAY = "#666666"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"


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
        "EEM_eCMR_LPP_EmotionalOnly",
        "predicted",
        "Emotion-based + LPP-based\nlearning boost for negative items only",
    ),
)
CATEGORIES = (
    ("Negative", CORAL),
    ("Neutral", DARK_GRAY),
)
CONTRAST_NAME = "Negative minus Neutral recall rate"

STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def load_selected_rows() -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the six plotted mean-and-interval rows."""
    fieldnames, rows = _read_csv(SUMMARY_SOURCE)
    source_by_id = {source.source_id: source for source in SOURCES}
    selected = [
        row
        for row in rows
        if row["source_id"] in source_by_id and row["metric"] == "recall_rate"
    ]
    expected = {
        (source.source_id, category)
        for source in SOURCES
        for category, _ in CATEGORIES
    }
    actual = [(row["source_id"], row["category"]) for row in selected]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate selected recall-rate rows")
    if set(actual) != expected or len(selected) != 6:
        raise ValueError(
            f"Selected recall rows do not match the expected six cells: {actual}"
        )

    for row in selected:
        source = source_by_id[row["source_id"]]
        if row["source_type"] != source.source_type:
            raise ValueError(f"Unexpected source type in row: {row}")
        mean, lower, upper = (float(row[key]) for key in ("mean", "lower", "upper"))
        if not np.isfinite([mean, lower, upper]).all() or not lower <= mean <= upper:
            raise ValueError(f"Invalid mean or interval in row: {row}")
        if float(row["confidence_level"]) != 0.95:
            raise ValueError(f"Unexpected interval level in row: {row}")
        if source.source_type == "observed":
            if row["interval"] != "Participant-cluster percentile bootstrap":
                raise ValueError(f"Unexpected observed interval: {row}")
            if int(row["n_units"]) != 38:
                raise ValueError(f"Unexpected observed sample size: {row}")
        else:
            if row["interval"] != "Central interval across complete simulated datasets":
                raise ValueError(f"Unexpected prediction interval: {row}")
            if int(row["n_units"]) != 200:
                raise ValueError(f"Unexpected prediction count: {row}")

    source_order = {source.source_id: index for index, source in enumerate(SOURCES)}
    category_order = {category: index for index, (category, _) in enumerate(CATEGORIES)}
    selected.sort(
        key=lambda row: (
            source_order[row["source_id"]],
            category_order[row["category"]],
        )
    )
    return fieldnames, selected


def load_contrasts() -> dict[str, float]:
    """Read the exact observed and predicted contrasts used in annotations."""
    _, rows = _read_csv(CONTRAST_SOURCE)
    source_ids = {source.source_id for source in SOURCES}
    selected = {
        row["source_id"]: float(row["estimate"])
        for row in rows
        if row["source_id"] in source_ids and row["contrast"] == CONTRAST_NAME
    }
    if set(selected) != source_ids:
        raise ValueError("Missing selected recall-rate contrasts")
    return selected


def write_summary(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with FILTERED_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def plot(rows: list[dict[str, str]], contrasts: dict[str, float]) -> None:
    lookup = {(row["source_id"], row["category"]): row for row in rows}
    positions = np.asarray([0.0, 0.92])
    colors = [color for _, color in CATEGORIES]
    category_labels = [category for category, _ in CATEGORIES]
    highest_label = 0.0

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1,
            len(SOURCES),
            figsize=(8.8, 3.95),
            sharey=True,
        )
        for panel_index, (axis, source) in enumerate(zip(axes, SOURCES, strict=True)):
            source_rows = [lookup[(source.source_id, category)] for category, _ in CATEGORIES]
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
                error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.1, "capsize": 3},
                zorder=3,
            )
            axis.text(
                0.5,
                1.23,
                source.title,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                linespacing=1.2,
                clip_on=False,
            )
            axis.set_xticks(positions, category_labels)
            axis.tick_params(axis="x", length=0, pad=4)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            if panel_index > 0:
                axis.tick_params(axis="y", labelleft=False)

            plotted_difference = float(means[0] - means[1])
            expected_difference = contrasts[source.source_id]
            if not np.isclose(plotted_difference, expected_difference, rtol=0, atol=1e-12):
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
        axes[0].set_ylim(0, 0.62)
        axes[0].set_ylabel("Recall rate", labelpad=7)
        figure.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.12, wspace=0.22)
        figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    fieldnames, rows = load_selected_rows()
    contrasts = load_contrasts()
    write_summary(fieldnames, rows)
    plot(rows, contrasts)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved {FILTERED_SUMMARY}")


if __name__ == "__main__":
    main()
