"""Plot mixed- and pure-list Early-LPP summaries for one fitted model."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = "EEM_eCMR_LPP_General"
WORK_DIR = Path(__file__).resolve().parent
INPUTS = {
    "Mixed lists": PROJECT_ROOT
    / "work"
    / "lpp_model_comparison"
    / f"{MODEL}_best_of_3.h5",
    "Pure lists": WORK_DIR / f"{MODEL}_best_of_3.h5",
}
OUTPUT_STEM = WORK_DIR / f"{MODEL}_mixed_pure_lpp_bars"

# Sampled from the four violin fills in the supplied empirical figure.
COLORS = {
    (1, True): "#ED706B",  # negative remembered
    (1, False): "#F4B5B4",  # negative forgotten
    (2, True): "#666666",  # neutral remembered
    (2, False): "#B3B3B3",  # neutral forgotten
}
CELLS = (
    (1, True, "Negative\nRecalled"),
    (1, False, "Negative\nForgotten"),
    (2, True, "Neutral\nRecalled"),
    (2, False, "Neutral\nForgotten"),
)


def load_simulation(path: Path) -> tuple[np.ndarray, ...]:
    """Load trial-major arrays from a jaxcmr HDF5 simulation."""
    with h5py.File(path) as handle:
        data = handle["data"]
        lpp = np.asarray(data["EarlyLPP"]).T
        condition = np.asarray(data["condition"]).T
        recalls = np.asarray(data["recalls"]).T
        subjects = np.asarray(data["subject"]).reshape(-1)
    return lpp, condition, recalls, subjects


def recalled_items(recalls: np.ndarray) -> np.ndarray:
    """Convert one-indexed recalled positions to an item-level mask."""
    recalled = np.zeros_like(recalls, dtype=bool)
    rows = np.repeat(np.arange(recalls.shape[0]), recalls.shape[1])
    positions = recalls.reshape(-1).astype(int)
    valid = positions > 0
    recalled[rows[valid], positions[valid] - 1] = True
    return recalled


def summarize(path: Path, list_type: str) -> list[dict[str, float | int | str]]:
    """Summarize each cell across participant-level simulated means."""
    lpp, condition, recalls, subjects = load_simulation(path)
    recalled = recalled_items(recalls)
    rows: list[dict[str, float | int | str]] = []

    for category, is_recalled, label in CELLS:
        participant_means = []
        for subject in np.unique(subjects):
            mask = (
                (subjects[:, np.newaxis] == subject)
                & (condition == category)
                & (recalled == is_recalled)
            )
            participant_means.append(float(np.mean(lpp[mask])))
        participant_means_array = np.asarray(participant_means)
        mean = float(np.mean(participant_means_array))
        sem = float(np.std(participant_means_array, ddof=1) / np.sqrt(len(participant_means_array)))
        rows.append(
            {
                "list_type": list_type,
                "condition": "Negative" if category == 1 else "Neutral",
                "recall": "Recalled" if is_recalled else "Forgotten",
                "label": label.replace("\n", " "),
                "mean": mean,
                "sem": sem,
                "ci95_low": mean - 1.96 * sem,
                "ci95_high": mean + 1.96 * sem,
                "participant_count": int(len(participant_means_array)),
            }
        )
    return rows


def add_difference_bracket(
    axis: plt.Axes, left: float, right: float, height: float, difference: float
) -> None:
    """Annotate a remembered-minus-forgotten model contrast."""
    tick = 0.025
    axis.plot(
        [left, left, right, right],
        [height - tick, height, height, height - tick],
        color="#333333",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        (left + right) / 2,
        height + 0.018,
        rf"$\Delta={difference:.2f}$",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )


def plot(rows: list[dict[str, float | int | str]]) -> None:
    """Draw shared-scale mixed- and pure-list bar plots."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 4.25), sharey=True)
    positions = np.array([0.0, 0.82, 2.05, 2.87])
    width = 0.62

    for panel_index, (axis, list_type) in enumerate(zip(axes, INPUTS, strict=True)):
        panel_rows = [row for row in rows if row["list_type"] == list_type]
        means = np.asarray([float(row["mean"]) for row in panel_rows])
        lower = means - np.asarray([float(row["ci95_low"]) for row in panel_rows])
        upper = np.asarray([float(row["ci95_high"]) for row in panel_rows]) - means
        colors = [COLORS[(category, recalled)] for category, recalled, _ in CELLS]

        axis.bar(
            positions,
            means,
            width=width,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
            yerr=np.vstack((lower, upper)),
            error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capsize": 3},
            zorder=3,
        )
        axis.set_title(list_type, pad=10)
        axis.set_xticks(positions, [label for _, _, label in CELLS])
        axis.set_xlabel("Condition")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
        axis.tick_params(axis="x", labelsize=8.5)
        axis.text(
            -0.13,
            1.04,
            chr(ord("A") + panel_index),
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        negative_difference = means[0] - means[1]
        neutral_difference = means[2] - means[3]
        negative_height = max(upper[0] + means[0], upper[1] + means[1]) + 0.11
        neutral_height = max(upper[2] + means[2], upper[3] + means[3]) + 0.11
        add_difference_bracket(
            axis, positions[0], positions[1], negative_height, negative_difference
        )
        add_difference_bracket(
            axis, positions[2], positions[3], neutral_height, neutral_difference
        )

    axes[0].set_ylabel("Early LPP amplitude (z)")
    axes[0].set_ylim(0, 1.42)
    figure.text(
        0.5,
        0.01,
        "Bars show participant-mean predictions; error bars are 95% CIs across 38 participant summaries.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1), w_pad=2.2)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    rows = [
        row
        for list_type, path in INPUTS.items()
        for row in summarize(path, list_type)
    ]
    summary_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
