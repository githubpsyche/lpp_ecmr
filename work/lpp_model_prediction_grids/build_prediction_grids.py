"""Build prediction-only diagnostic grids for the six focal eCMR models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
SUMMARY_SOURCE = PROJECT_ROOT / "work" / "lpp_model_results" / "diagnostic_summary.csv"
CONTRAST_SOURCE = (
    PROJECT_ROOT / "work" / "lpp_model_results" / "diagnostic_contrasts.csv"
)
FILTERED_SUMMARY = WORK_DIR / "prediction_summary.csv"

RECALL_OUTPUT = WORK_DIR / "recall_rate_prediction_grid"
LPP_OUTPUT = WORK_DIR / "early_lpp_prediction_grid"

CORAL = "#ED706B"
PALE_PINK = "#F4B5B4"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#B3B3B3"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class ModelCell:
    registry_id: str
    emotion_effect: str
    lpp_effect: str


MODEL_GRID = (
    (
        ModelCell("CategoryOnly_eCMR", "absent", "none"),
        ModelCell("CategoryOnly_eCMR_LPP_General", "absent", "general"),
        ModelCell(
            "CategoryOnly_eCMR_LPP_EmotionalOnly",
            "absent",
            "emotion-specific",
        ),
    ),
    (
        ModelCell("EEM_eCMR", "present", "none"),
        ModelCell("EEM_eCMR_LPP_General", "present", "general"),
        ModelCell(
            "EEM_eCMR_LPP_EmotionalOnly",
            "present",
            "emotion-specific",
        ),
    ),
)
MODEL_IDS = {cell.registry_id for row in MODEL_GRID for cell in row}

COLUMN_LABELS = (
    "No LPP effect",
    "General LPP effect",
    "Emotion-specific\nLPP effect",
)
ROW_LABELS = ("Emotion effect\nabsent", "Emotion effect\npresent")

RECALL_CELLS = (
    ("Negative", "", "Negative", CORAL),
    ("Neutral", "", "Neutral", DARK_GRAY),
)
LPP_CELLS = (
    ("Negative", "Remembered", "Negative\nRemembered", CORAL),
    ("Negative", "Forgotten", "Negative\nForgotten", PALE_PINK),
    ("Neutral", "Remembered", "Neutral\nRemembered", DARK_GRAY),
    ("Neutral", "Forgotten", "Neutral\nForgotten", LIGHT_GRAY),
)

STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _expected_keys() -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for model_id in MODEL_IDS:
        keys.update(
            (model_id, "recall_rate", category, recall_status)
            for category, recall_status, _, _ in RECALL_CELLS
        )
        keys.update(
            (model_id, "within_list_centered_early_lpp", category, recall_status)
            for category, recall_status, _, _ in LPP_CELLS
        )
    return keys


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["source_id"],
        row["metric"],
        row["category"],
        row["recall_status"],
    )


def load_predictions() -> tuple[list[str], list[dict[str, str]]]:
    """Load and validate the six prediction summaries used by both figures."""
    fieldnames, rows = _read_csv(SUMMARY_SOURCE)
    prediction_rows = [row for row in rows if row["source_type"] == "predicted"]
    prediction_ids = {row["source_id"] for row in prediction_rows}

    unexpected = prediction_ids - MODEL_IDS
    missing = MODEL_IDS - prediction_ids
    if unexpected or missing:
        raise ValueError(
            f"Prediction model mismatch; unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )

    selected = [row for row in prediction_rows if row["source_id"] in MODEL_IDS]
    expected_keys = _expected_keys()
    selected_keys = [_row_key(row) for row in selected]
    duplicate_keys = {key for key in selected_keys if selected_keys.count(key) > 1}
    if duplicate_keys:
        raise ValueError(f"Duplicate diagnostic rows: {sorted(duplicate_keys)}")
    if set(selected_keys) != expected_keys:
        missing_keys = expected_keys - set(selected_keys)
        unexpected_keys = set(selected_keys) - expected_keys
        raise ValueError(
            f"Diagnostic-row mismatch; unexpected={sorted(unexpected_keys)}, "
            f"missing={sorted(missing_keys)}"
        )

    recall_rows = [row for row in selected if row["metric"] == "recall_rate"]
    lpp_rows = [
        row
        for row in selected
        if row["metric"] == "within_list_centered_early_lpp"
    ]
    if len(selected) != 36 or len(recall_rows) != 12 or len(lpp_rows) != 24:
        raise ValueError(
            "Expected 36 prediction rows: 12 recall-rate and 24 Early-LPP rows; "
            f"found {len(selected)}, {len(recall_rows)}, and {len(lpp_rows)}"
        )

    for row in selected:
        mean = float(row["mean"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        if not np.isfinite([mean, lower, upper]).all() or not lower <= mean <= upper:
            raise ValueError(f"Invalid interval in diagnostic row: {row}")
        if row["interval"] != "Central interval across complete simulated datasets":
            raise ValueError(f"Unexpected interval definition: {row['interval']}")
        if float(row["confidence_level"]) != 0.95 or int(row["n_units"]) != 200:
            raise ValueError(f"Unexpected prediction-interval metadata: {row}")

    model_order = {
        cell.registry_id: row_index * len(MODEL_GRID[0]) + column_index
        for row_index, model_row in enumerate(MODEL_GRID)
        for column_index, cell in enumerate(model_row)
    }
    metric_order = {"recall_rate": 0, "within_list_centered_early_lpp": 1}
    category_order = {"Negative": 0, "Neutral": 1}
    recall_order = {"": 0, "Remembered": 0, "Forgotten": 1}
    selected.sort(
        key=lambda row: (
            model_order[row["source_id"]],
            metric_order[row["metric"]],
            category_order[row["category"]],
            recall_order[row["recall_status"]],
        )
    )
    return fieldnames, selected


def write_filtered_summary(
    fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> None:
    with FILTERED_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_lookup(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {_row_key(row): row for row in rows}


def load_contrasts() -> dict[tuple[str, str], float]:
    _, rows = _read_csv(CONTRAST_SOURCE)
    contrasts = {
        (row["source_id"], row["contrast"]): float(row["estimate"])
        for row in rows
        if row["source_type"] == "predicted" and row["source_id"] in MODEL_IDS
    }
    expected_names = {
        "Negative minus Neutral recall rate",
        "Remembered minus Forgotten centered Early LPP: Negative",
        "Remembered minus Forgotten centered Early LPP: Neutral",
    }
    expected = {(model_id, name) for model_id in MODEL_IDS for name in expected_names}
    if set(contrasts) != expected:
        raise ValueError("The diagnostic-contrast source does not match the six-model grid")
    return contrasts


def validate_delta(
    contrasts: dict[tuple[str, str], float],
    model_id: str,
    contrast_name: str,
    difference: float,
) -> None:
    expected = contrasts[(model_id, contrast_name)]
    if not np.isclose(difference, expected, rtol=0, atol=1e-12):
        raise AssertionError(
            f"Delta mismatch for {model_id}, {contrast_name}: "
            f"plotted={difference}, expected={expected}"
        )


def add_difference_bracket(
    axis: plt.Axes,
    left: float,
    right: float,
    height: float,
    difference: float,
    *,
    tick: float,
    label_offset: float,
) -> float:
    """Annotate a difference between two displayed means."""
    axis.plot(
        [left, left, right, right],
        [height - tick, height, height, height - tick],
        color=EDGE_COLOR,
        linewidth=0.9,
        clip_on=False,
    )
    label_height = height + label_offset
    rounded_difference = round(difference, 2)
    if rounded_difference == 0:
        rounded_difference = 0.0
    axis.text(
        (left + right) / 2,
        label_height,
        rf"$\Delta={rounded_difference:.2f}$",
        ha="center",
        va="bottom",
        fontsize=9,
        color=EDGE_COLOR,
    )
    return label_height


def make_grid() -> tuple[plt.Figure, np.ndarray]:
    figure = plt.figure(figsize=(12, 7.5))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.34),
        wspace=0.18,
        hspace=0.52,
    )
    axes = np.empty((2, 3), dtype=object)
    shared_axis: plt.Axes | None = None

    for row_index, row_label in enumerate(ROW_LABELS):
        for column_index in range(3):
            axis = figure.add_subplot(
                grid[row_index, column_index],
                sharey=shared_axis,
            )
            if shared_axis is None:
                shared_axis = axis
            axes[row_index, column_index] = axis
            if row_index == 0:
                axis.set_title(COLUMN_LABELS[column_index], pad=13, fontsize=11)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            axis.tick_params(axis="x", length=0, labelsize=8.5)
            if column_index > 0:
                axis.tick_params(axis="y", labelleft=False)

        label_axis = figure.add_subplot(grid[row_index, 3])
        label_axis.axis("off")
        label_axis.text(
            0.04,
            0.5,
            row_label,
            ha="left",
            va="center",
            fontsize=11,
            linespacing=1.35,
        )

    return figure, axes


def draw_bars(
    axis: plt.Axes,
    positions: np.ndarray,
    width: float,
    labels: list[str],
    colors: list[str],
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray([float(row["mean"]) for row in rows])
    lower_bounds = np.asarray([float(row["lower"]) for row in rows])
    upper_bounds = np.asarray([float(row["upper"]) for row in rows])
    lower_errors = means - lower_bounds
    upper_errors = upper_bounds - means

    axis.bar(
        positions,
        means,
        width=width,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        yerr=np.vstack((lower_errors, upper_errors)),
        error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.1, "capsize": 3},
        zorder=3,
    )
    axis.set_xticks(positions, labels)
    return means, lower_bounds, upper_bounds


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_recall_rate(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    contrasts: dict[tuple[str, str], float],
) -> None:
    positions = np.asarray([0.0, 0.95])
    labels = [cell[2] for cell in RECALL_CELLS]
    colors = [cell[3] for cell in RECALL_CELLS]
    highest_annotation = 0.0

    with plt.rc_context(STYLE):
        figure, axes = make_grid()
        for row_index, model_row in enumerate(MODEL_GRID):
            for column_index, model in enumerate(model_row):
                axis = axes[row_index, column_index]
                rows = [
                    lookup[(model.registry_id, "recall_rate", category, recall_status)]
                    for category, recall_status, _, _ in RECALL_CELLS
                ]
                means, _, upper_bounds = draw_bars(
                    axis,
                    positions,
                    0.65,
                    labels,
                    colors,
                    rows,
                )
                difference = float(means[0] - means[1])
                validate_delta(
                    contrasts,
                    model.registry_id,
                    "Negative minus Neutral recall rate",
                    difference,
                )
                label_height = add_difference_bracket(
                    axis,
                    positions[0],
                    positions[1],
                    float(np.max(upper_bounds) + 0.025),
                    difference,
                    tick=0.012,
                    label_offset=0.008,
                )
                highest_annotation = max(highest_annotation, label_height)

        if highest_annotation >= 0.62:
            raise AssertionError("Recall-rate annotation exceeds the shared y limit")
        axes[0, 0].set_ylim(0, 0.62)
        figure.supylabel("Recall rate", x=0.034, fontsize=11)
        figure.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.08)
        save_figure(figure, RECALL_OUTPUT)


def plot_early_lpp(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    contrasts: dict[tuple[str, str], float],
) -> None:
    positions = np.asarray([0.0, 1.0, 2.35, 3.35])
    labels = ["Remembered", "Forgotten", "Remembered", "Forgotten"]
    colors = [cell[3] for cell in LPP_CELLS]
    highest_annotation = -np.inf
    lowest_interval = np.inf

    with plt.rc_context(STYLE):
        figure, axes = make_grid()
        for row_index, model_row in enumerate(MODEL_GRID):
            for column_index, model in enumerate(model_row):
                axis = axes[row_index, column_index]
                rows = [
                    lookup[
                        (
                            model.registry_id,
                            "within_list_centered_early_lpp",
                            category,
                            recall_status,
                        )
                    ]
                    for category, recall_status, _, _ in LPP_CELLS
                ]
                means, lower_bounds, upper_bounds = draw_bars(
                    axis,
                    positions,
                    0.62,
                    labels,
                    colors,
                    rows,
                )
                axis.axhline(0, color="#888888", linewidth=0.9, zorder=1)
                axis.tick_params(axis="x", labelsize=8)
                axis.set_xticklabels(
                    labels,
                    rotation=35,
                    ha="right",
                    rotation_mode="anchor",
                )
                axis.text(
                    float(np.mean(positions[:2])),
                    -0.28,
                    "Negative",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    clip_on=False,
                )
                axis.text(
                    float(np.mean(positions[2:])),
                    -0.28,
                    "Neutral",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    clip_on=False,
                )
                lowest_interval = min(lowest_interval, float(np.min(lower_bounds)))

                negative_difference = float(means[0] - means[1])
                neutral_difference = float(means[2] - means[3])
                validate_delta(
                    contrasts,
                    model.registry_id,
                    "Remembered minus Forgotten centered Early LPP: Negative",
                    negative_difference,
                )
                validate_delta(
                    contrasts,
                    model.registry_id,
                    "Remembered minus Forgotten centered Early LPP: Neutral",
                    neutral_difference,
                )

                negative_label_height = add_difference_bracket(
                    axis,
                    positions[0],
                    positions[1],
                    float(np.max(upper_bounds[:2]) + 0.06),
                    negative_difference,
                    tick=0.025,
                    label_offset=0.018,
                )
                neutral_label_height = add_difference_bracket(
                    axis,
                    positions[2],
                    positions[3],
                    float(np.max(upper_bounds[2:]) + 0.06),
                    neutral_difference,
                    tick=0.025,
                    label_offset=0.018,
                )
                highest_annotation = max(
                    highest_annotation,
                    negative_label_height,
                    neutral_label_height,
                )

        if lowest_interval <= -0.32 or highest_annotation >= 0.72:
            raise AssertionError(
                "Early-LPP intervals or annotations exceed the shared y limits"
            )
        axes[0, 0].set_ylim(-0.32, 0.72)
        figure.supylabel(
            "Mean within-list-centered Early LPP (z)",
            x=0.034,
            fontsize=11,
        )
        figure.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.17)
        save_figure(figure, LPP_OUTPUT)


def main() -> None:
    fieldnames, rows = load_predictions()
    write_filtered_summary(fieldnames, rows)
    lookup = build_lookup(rows)
    contrasts = load_contrasts()
    plot_recall_rate(lookup, contrasts)
    plot_early_lpp(lookup, contrasts)
    print(f"Saved {RECALL_OUTPUT.with_suffix('.png')}")
    print(f"Saved {RECALL_OUTPUT.with_suffix('.svg')}")
    print(f"Saved {LPP_OUTPUT.with_suffix('.png')}")
    print(f"Saved {LPP_OUTPUT.with_suffix('.svg')}")
    print(f"Saved {FILTERED_SUMMARY}")


if __name__ == "__main__":
    main()
