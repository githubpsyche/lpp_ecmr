"""Overlay observed targets on selected model Early-LPP contrast bars."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


WORK_DIR = Path(__file__).resolve().parent
CONTRAST_PATH = WORK_DIR / "original_early_lpp_contrasts.csv"
OUTPUT_STEM = WORK_DIR / "selected_early_lpp_target_overlay"

CORAL = "#ED706B"
DARK_GRAY = "#666666"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
OBSERVED_BAND_COLOR = "#A6A6A6"

MODELS = (
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

    source_ids = {"Observed", *(model_id for model_id, _ in MODELS)}
    expected = {
        (source_id, category)
        for source_id in source_ids
        for category, _ in CATEGORIES
    }
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    prefix = "Remembered minus Forgotten Early LPP: "
    for row in rows:
        if row["source_id"] not in source_ids or not row["contrast"].startswith(prefix):
            continue
        key = (row["source_id"], row["contrast"].removeprefix(prefix))
        if key in lookup:
            raise ValueError(f"Duplicate contrast row: {key}")
        values = {
            name: float(row[name]) for name in ("estimate", "lower", "upper")
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Non-finite contrast row: {row}")
        if not values["lower"] <= values["estimate"] <= values["upper"]:
            raise ValueError(f"Invalid contrast interval: {row}")
        lookup[key] = values

    if set(lookup) != expected:
        raise ValueError(
            f"Target-overlay rows do not match expectation; "
            f"missing={expected - set(lookup)}, extra={set(lookup) - expected}"
        )
    return lookup


def plot(lookup: dict[tuple[str, str], dict[str, float]]) -> None:
    all_rows = list(lookup.values())
    lower = min(row["lower"] for row in all_rows)
    upper = max(row["upper"] for row in all_rows)
    value_range = upper - lower
    y_limits = (
        min(-0.16, lower - 0.04 * value_range),
        upper + 0.09 * value_range,
    )
    positions = np.asarray([0.0, 1.0])
    observed_band_width = 0.80
    bar_width = 0.52

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1,
            len(MODELS),
            figsize=(10.2, 4.2),
            sharey=True,
        )
        for panel_index, (axis, (model_id, title)) in enumerate(
            zip(axes, MODELS, strict=True)
        ):
            observed_rows = [
                lookup[("Observed", category)] for category, _ in CATEGORIES
            ]
            model_rows = [lookup[(model_id, category)] for category, _ in CATEGORIES]

            for position, observed in zip(positions, observed_rows, strict=True):
                axis.add_patch(
                    Rectangle(
                        (
                            position - observed_band_width / 2,
                            observed["lower"],
                        ),
                        observed_band_width,
                        observed["upper"] - observed["lower"],
                        facecolor=OBSERVED_BAND_COLOR,
                        edgecolor="none",
                        alpha=0.24,
                        zorder=1,
                    )
                )

            estimates = np.asarray([row["estimate"] for row in model_rows])
            model_lower = np.asarray([row["lower"] for row in model_rows])
            model_upper = np.asarray([row["upper"] for row in model_rows])
            model_errors = np.vstack(
                (estimates - model_lower, model_upper - estimates)
            )
            axis.bar(
                positions,
                estimates,
                width=bar_width,
                color=[color for _, color in CATEGORIES],
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                yerr=model_errors,
                error_kw={
                    "ecolor": ERROR_COLOR,
                    "elinewidth": 1.1,
                    "capsize": 3.0,
                },
                zorder=3,
            )

            for position, observed in zip(positions, observed_rows, strict=True):
                axis.hlines(
                    observed["estimate"],
                    position - observed_band_width / 2,
                    position + observed_band_width / 2,
                    color=EDGE_COLOR,
                    linewidth=2.0,
                    zorder=5,
                )

            axis.set_title(title, fontsize=10.3, pad=10, linespacing=1.18)
            axis.set_xticks(positions, [category for category, _ in CATEGORIES])
            axis.tick_params(axis="x", length=0, pad=5)
            axis.set_xlim(-0.55, 1.55)
            axis.set_ylim(*y_limits)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
            axis.axhline(0, color="#7F7F7F", linewidth=0.9, zorder=2)
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
            if panel_index > 0:
                axis.tick_params(axis="y", labelleft=False)

        legend_handles = (
            Patch(
                facecolor="#808080",
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                label="Model prediction",
            ),
            Line2D(
                [0],
                [0],
                color=EDGE_COLOR,
                linewidth=2.0,
                label="Observed",
            ),
            Patch(
                facecolor=OBSERVED_BAND_COLOR,
                edgecolor="none",
                alpha=0.24,
                label="Observed 95% CI",
            ),
        )
        figure.supylabel(
            "Remembered − forgotten Early LPP (z)",
            x=0.025,
            fontsize=10.5,
        )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
            handlelength=1.7,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        figure.subplots_adjust(
            left=0.085,
            right=0.99,
            top=0.78,
            bottom=0.22,
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
