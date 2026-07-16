"""Simulate and plot mixed/pure LPP summaries for a hand-calibrated candidate.

The candidate starts from the pooled EEM_eCMR_LPP_General fit and changes only
``emotion_scale`` from its fitted value to 1.3. It is a model demonstration,
not a refit or a maximum-likelihood estimate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jaxcmr.components.context import init as init_context
from jaxcmr.components.linear_memory import init_mcf, init_mfc
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import generate_trial_mask, load_data
from jaxcmr.simulation import (
    simulate_h5_from_h5,
    simulate_study_free_recall_and_forced_stop,
)

from lpp_ecmr.models.full_eeg_ecmr import make_factory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
MODEL = "EEM_eCMR_LPP_General"
SOURCE_FIT = (
    PROJECT_ROOT / "work" / "lpp_model_comparison" / f"{MODEL}_best_of_3.json"
)
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"
PURE_DESIGN_PATH = WORK_DIR / "pure_list_design.csv"
EMOTION_SCALE = 1.3
EXPERIMENT_COUNT = 200
SEEDS = {"Mixed lists": 9200, "Pure lists": 9201}
OUTPUT_STEM = WORK_DIR / (
    f"{MODEL}_emotion_scale_1p3_mixed_pure_lpp_bars"
)

# Sampled from the four violin fills in the supplied empirical figure.
COLORS = {
    (1, True): "#ED706B",  # negative recalled
    (1, False): "#F4B5B4",  # negative forgotten
    (2, True): "#666666",  # neutral recalled
    (2, False): "#B3B3B3",  # neutral forgotten
}
CELLS = (
    (1, True, "Negative\nRecalled"),
    (1, False, "Negative\nForgotten"),
    (2, True, "Neutral\nRecalled"),
    (2, False, "Neutral\nForgotten"),
)


def load_candidate_parameters() -> tuple[dict[str, jnp.ndarray], dict[str, object]]:
    """Load the pooled fit and replace only its categorical emotion multiplier."""
    with SOURCE_FIT.open() as handle:
        fit = json.load(handle)

    fitted_emotion_scale = float(fit["fits"]["emotion_scale"][0])
    candidate_fits = {key: list(values) for key, values in fit["fits"].items()}
    candidate_fits["emotion_scale"] = [EMOTION_SCALE]
    params = {key: jnp.asarray(values) for key, values in candidate_fits.items()}

    config: dict[str, object] = {
        "status": "hand-calibrated model demonstration; not a refit",
        "source_fit": str(SOURCE_FIT.relative_to(PROJECT_ROOT)),
        "model": MODEL,
        "experiment_count": EXPERIMENT_COUNT,
        "seeds": SEEDS,
        "override": {
            "parameter": "emotion_scale",
            "fitted_value": fitted_emotion_scale,
            "candidate_value": EMOTION_SCALE,
        },
        "fits": candidate_fits,
    }
    return params, config


def make_pure_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Relabel each original list as pure negative or pure neutral."""
    with PURE_DESIGN_PATH.open(newline="") as handle:
        design = list(csv.DictReader(handle))

    trial_count = np.asarray(data["subject"]).shape[0]
    trial_indices = np.asarray(
        [int(row["trial_index"]) for row in design], dtype=np.int32
    )
    pure_conditions = np.asarray(
        [int(row["condition"]) for row in design], dtype=np.int32
    )
    if not np.array_equal(trial_indices, np.arange(trial_count)):
        raise ValueError("pure_list_design.csv does not match the data trial order")
    if np.count_nonzero(pure_conditions == 1) != trial_count // 2:
        raise ValueError("Expected half of the lists to be pure negative")
    if np.count_nonzero(pure_conditions == 2) != trial_count // 2:
        raise ValueError("Expected half of the lists to be pure neutral")

    simulation_fields = (
        "EarlyLPP",
        "list",
        "listLength",
        "pres_itemids",
        "pres_itemnos",
        "recalls",
        "subject",
    )
    pure_data = {key: np.array(data[key], copy=True) for key in simulation_fields}
    pure_data["condition"] = np.repeat(
        pure_conditions[:, np.newaxis],
        np.asarray(data["condition"]).shape[1],
        axis=1,
    ).astype(np.int32)
    pure_data["list_type"] = pure_conditions[:, np.newaxis]
    return pure_data


def recalled_items(recalls: np.ndarray) -> np.ndarray:
    """Convert one-indexed recalled positions to an item-level mask."""
    recalled = np.zeros_like(recalls, dtype=bool)
    rows = np.repeat(np.arange(recalls.shape[0]), recalls.shape[1])
    positions = recalls.reshape(-1).astype(int)
    valid = positions > 0
    recalled[rows[valid], positions[valid] - 1] = True
    return recalled


def summarize(
    sim: dict[str, np.ndarray], list_type: str
) -> list[dict[str, float | int | str]]:
    """Summarize each cell across participant-level simulated means."""
    lpp = np.asarray(sim["EarlyLPP"])
    condition = np.asarray(sim["condition"])
    recalled = recalled_items(np.asarray(sim["recalls"]))
    subjects = np.asarray(sim["subject"]).reshape(-1)
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
        sem = float(
            np.std(participant_means_array, ddof=1)
            / np.sqrt(len(participant_means_array))
        )
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


def simulate_and_summarize(
    data: dict[str, np.ndarray],
    parameters: dict[str, jnp.ndarray],
    list_type: str,
) -> list[dict[str, float | int | str]]:
    """Run one composition through the candidate model and summarize it."""
    factory = make_factory(
        mfc_create_fn=init_mfc,
        mcf_create_fn=init_mcf,
        context_create_fn=init_context,
        termination_policy_create_fn=NoStopTermination,
    )
    unique_subjects = jnp.unique(jnp.asarray(data["subject"]))
    subject_parameters = {
        key: jnp.repeat(value, unique_subjects.shape[0])
        if key != "subject"
        else unique_subjects
        for key, value in parameters.items()
    }
    trial_mask = generate_trial_mask(data, "data['subject'] > -1")
    sim = simulate_h5_from_h5(
        factory,
        data,
        None,
        subject_parameters,
        trial_mask,
        EXPERIMENT_COUNT,
        random.PRNGKey(SEEDS[list_type]),
        simulate_trial_fn=simulate_study_free_recall_and_forced_stop,
    )
    return summarize(sim, list_type)


def add_difference_bracket(
    axis: plt.Axes, left: float, right: float, height: float, difference: float
) -> None:
    """Annotate a recalled-minus-forgotten model contrast."""
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

    for panel_index, (axis, list_type) in enumerate(
        zip(axes, SEEDS, strict=True)
    ):
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
    data = load_data(DATA_PATH)
    parameters, config = load_candidate_parameters()
    rows = simulate_and_summarize(data, parameters, "Mixed lists")
    rows.extend(
        simulate_and_summarize(make_pure_data(data), parameters, "Pure lists")
    )

    summary_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_config.json")
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)

    plot(rows)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved {summary_path}")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
