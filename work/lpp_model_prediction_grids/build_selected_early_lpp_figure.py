"""Build the selected empirical-versus-predicted Early-LPP comparison.

The observed panel follows the participant-level summarization in Robin
Hellerstedt's mixed-list R analysis and reads its original z-transformed
single-trial Early-LPP data directly from the downloaded archive. Predicted
panels use the returned simulations but summarize the uncentered ``EarlyLPP``
input rather than the within-list-centered predictor used during fitting.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
FIT_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
DOWNLOAD_ARCHIVE = (
    PROJECT_ROOT.parent
    / "downloads"
    / "LPP_Zarubin_PureLists_for_Deborah.zip"
)
EMPIRICAL_DATA_MEMBER = (
    "Data/Extracted_Behavioural_and_LPP_Data/"
    "Single_Trial_Behavioural_and_EEG_Data_Z.csv"
)
EMPIRICAL_CODE_MEMBER = "R_Script/Behaviour_and_Relationship_to_LPP.Rmd"

PARTICIPANT_MEANS_PATH = WORK_DIR / "empirical_early_lpp_participant_means.csv"
SUMMARY_PATH = WORK_DIR / "original_early_lpp_summary.csv"
CONTRAST_PATH = WORK_DIR / "original_early_lpp_contrasts.csv"
SOURCE_PATH = WORK_DIR / "original_early_lpp_source.json"
OUTPUT_STEM = WORK_DIR / "selected_early_lpp_figure"

BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
EXPECTED_PARTICIPANTS = 38
EXPECTED_LISTS = 342
EXPECTED_SIMULATIONS = 200
EXPECTED_EMPIRICAL_ROWS = 6_508
EXPECTED_OUT_OF_RANGE_PRESENTATION_ROWS = 39

NEGATIVE = 1
NEUTRAL = 2
CATEGORIES = (("Negative", NEGATIVE), ("Neutral", NEUTRAL))
MEMORY_STATUSES = (("Remembered", True), ("Forgotten", False))

CORAL = "#ED706B"
PALE_PINK = "#F4B5B4"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#B3B3B3"
EDGE_COLOR = "#333333"
ERROR_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"

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
        "EEM_eCMR",
        "predicted",
        "Emotion-based\nlearning boost",
    ),
    FigureSource(
        "EEM_eCMR_LPP_General",
        "predicted",
        "Emotion-based + LPP-based\nlearning boost for all items",
    ),
    FigureSource(
        "EEM_eCMR_LPP_EmotionalOnly",
        "predicted",
        "Emotion-based + LPP-based\nlearning boost for negative items only",
    ),
)

ALL_MODEL_IDS = (
    "CategoryOnly_eCMR",
    "CategoryOnly_eCMR_LPP_General",
    "CategoryOnly_eCMR_LPP_EmotionalOnly",
    "EEM_eCMR",
    "EEM_eCMR_LPP_General",
    "EEM_eCMR_LPP_EmotionalOnly",
)

SOURCE_LABELS = {
    "Observed": "Observed data",
    "CategoryOnly_eCMR": "Baseline",
    "CategoryOnly_eCMR_LPP_General": "LPP-based boost for all items",
    "CategoryOnly_eCMR_LPP_EmotionalOnly": (
        "LPP-based boost for negative items only"
    ),
    "EEM_eCMR": "Emotion-based boost",
    "EEM_eCMR_LPP_General": (
        "Emotion-based boost + LPP-based boost for all items"
    ),
    "EEM_eCMR_LPP_EmotionalOnly": (
        "Emotion-based boost + LPP-based boost for negative items only"
    ),
}

COLORS = (CORAL, PALE_PINK, DARK_GRAY, LIGHT_GRAY)
POSITIONS = np.asarray([0.0, 0.88, 2.18, 3.06])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_empirical_participant_means() -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Reproduce the R analysis's participant-by-cell Early-LPP means."""
    if not DOWNLOAD_ARCHIVE.exists():
        raise FileNotFoundError(
            "Missing empirical archive: "
            f"{DOWNLOAD_ARCHIVE}. The figure requires the original mixed-list "
            "z-LPP data downloaded on 2026-07-15."
        )

    with zipfile.ZipFile(DOWNLOAD_ARCHIVE) as archive:
        names = set(archive.namelist())
        required = {EMPIRICAL_DATA_MEMBER, EMPIRICAL_CODE_MEMBER}
        if not required.issubset(names):
            raise FileNotFoundError(
                f"Empirical archive is missing members: {sorted(required - names)}"
            )
        empirical_bytes = archive.read(EMPIRICAL_DATA_MEMBER)
        analysis_bytes = archive.read(EMPIRICAL_CODE_MEMBER)

    header = (
        "Participant",
        "List",
        "ThreeConds",
        "Condition",
        "TrialNumber",
        "PresentationOrder",
        "ImgName",
        "Memory",
        "EarlyLPP",
        "LateLPP",
    )
    reader = csv.DictReader(
        io.StringIO(empirical_bytes.decode("utf-8-sig")),
        fieldnames=header,
    )
    rows = list(reader)
    if len(rows) != EXPECTED_EMPIRICAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_EMPIRICAL_ROWS} empirical rows, found {len(rows)}"
        )

    cell_values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    out_of_range = 0
    for row in rows:
        participant = int(row["Participant"])
        condition = row["Condition"]
        if condition not in {"Negative", "Neutral"}:
            raise ValueError(f"Unexpected empirical condition: {condition}")
        memory = int(row["Memory"])
        if memory not in {0, 1}:
            raise ValueError(f"Unexpected empirical memory code: {memory}")
        status = "Remembered" if memory == 1 else "Forgotten"
        value = float(row["EarlyLPP"])
        if not np.isfinite(value):
            raise ValueError("Empirical Early-LPP data contain a non-finite value")
        presentation_order = int(float(row["PresentationOrder"]))
        out_of_range += int(not 1 <= presentation_order <= 20)
        cell_values[(participant, condition, status)].append(value)

    participants = sorted({key[0] for key in cell_values})
    if len(participants) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, found {len(participants)}"
        )
    if out_of_range != EXPECTED_OUT_OF_RANGE_PRESENTATION_ROWS:
        raise ValueError(
            "Unexpected count of empirical rows with non-model presentation order: "
            f"{out_of_range}"
        )

    means = np.full((len(participants), 2, 2), np.nan)
    output_rows: list[dict[str, object]] = []
    for participant_index, participant in enumerate(participants):
        for category_index, (category, _) in enumerate(CATEGORIES):
            for memory_index, (status, _) in enumerate(MEMORY_STATUSES):
                values = cell_values[(participant, category, status)]
                if not values:
                    raise ValueError(
                        f"Empty empirical cell: {participant}, {category}, {status}"
                    )
                mean = float(np.mean(values))
                means[participant_index, category_index, memory_index] = mean
                output_rows.append(
                    {
                        "participant": participant,
                        "category": category,
                        "recall_status": status,
                        "mean_early_lpp_z": mean,
                        "n_trials": len(values),
                    }
                )

    if not np.isfinite(means).all() or len(output_rows) != 152:
        raise ValueError("Empirical participant summaries are incomplete")

    metadata = {
        "archive": str(DOWNLOAD_ARCHIVE),
        "archive_sha256": _sha256(DOWNLOAD_ARCHIVE),
        "data_member": EMPIRICAL_DATA_MEMBER,
        "data_member_sha256": _bytes_sha256(empirical_bytes),
        "analysis_member": EMPIRICAL_CODE_MEMBER,
        "analysis_member_sha256": _bytes_sha256(analysis_bytes),
        "empirical_rows": len(rows),
        "participants": len(participants),
        "participant_cell_means": len(output_rows),
        "rows_with_presentation_order_outside_1_to_20": out_of_range,
    }
    return means, output_rows, metadata


def _recall_hits(recalls: np.ndarray, presentations: np.ndarray) -> np.ndarray:
    hits = np.zeros_like(presentations, dtype=bool)
    for event in range(recalls.shape[0]):
        recalled_position = recalls[event]
        hits |= (recalled_position[None, :] > 0) & (
            presentations == recalled_position[None, :]
        )
    return hits


def load_simulation_summary(model_id: str) -> tuple[np.ndarray, int]:
    """Return participant-averaged cell means for each simulated dataset."""
    path = FIT_DIR / f"{model_id}_best_of_3.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        group = handle["data"]
        data = {
            key: np.asarray(group[key][:])
            for key in (
                "subject",
                "condition",
                "pres_itemnos",
                "recalls",
                "EarlyLPP",
                "lpp_imputed",
            )
        }

    expected_columns = EXPECTED_LISTS * EXPECTED_SIMULATIONS
    if data["condition"].shape != (20, expected_columns):
        raise ValueError(f"Unexpected simulation shape in {path}")
    if data["subject"].shape != (1, expected_columns):
        raise ValueError(f"Unexpected subject shape in {path}")
    for key in ("pres_itemnos", "recalls", "EarlyLPP", "lpp_imputed"):
        if data[key].shape != (20, expected_columns):
            raise ValueError(f"Unexpected {key} shape in {path}: {data[key].shape}")

    def reshape(values: np.ndarray) -> np.ndarray:
        return values.reshape(values.shape[0], EXPECTED_LISTS, EXPECTED_SIMULATIONS)

    conditions = reshape(data["condition"])
    presentations = reshape(data["pres_itemnos"])
    lpp = reshape(data["EarlyLPP"].astype(float))
    lpp_imputed = reshape(data["lpp_imputed"])
    subjects = reshape(data["subject"])[0]
    hits = reshape(_recall_hits(data["recalls"], data["pres_itemnos"]))

    for name, values in (
        ("condition", conditions),
        ("presentation", presentations),
        ("EarlyLPP", lpp),
        ("lpp_imputed", lpp_imputed),
        ("subject", subjects),
    ):
        reference = values[..., :1]
        matches = (
            np.allclose(values, reference, equal_nan=True)
            if np.issubdtype(values.dtype, np.floating)
            else bool(np.all(values == reference))
        )
        if not matches:
            raise ValueError(f"Simulation storage-order check failed for {name}: {path}")

    participant_ids = np.unique(subjects[:, 0])
    if len(participant_ids) != EXPECTED_PARTICIPANTS:
        raise ValueError(f"Unexpected participant count in {path}")

    # The empirical plot excludes missing EEG observations. Do the same for the
    # model diagnostic rather than treating imputed fitting inputs as observed
    # EEG outcomes.
    valid = (presentations > 0) & (lpp_imputed == 0)
    nonimputed_positions = int(np.sum(valid[:, :, 0]))
    if nonimputed_positions != 6_469:
        raise ValueError(
            f"Expected 6,469 non-imputed model positions, found {nonimputed_positions}"
        )

    simulation_means = np.full((EXPECTED_SIMULATIONS, 2, 2), np.nan)
    for simulation in range(EXPECTED_SIMULATIONS):
        participant_means = np.full((EXPECTED_PARTICIPANTS, 2, 2), np.nan)
        for participant_index, participant_id in enumerate(participant_ids):
            list_mask = subjects[:, simulation] == participant_id
            for category_index, (_, category_code) in enumerate(CATEGORIES):
                category_mask = valid[:, :, simulation] & (
                    conditions[:, :, simulation] == category_code
                )
                category_mask &= list_mask[None, :]
                for memory_index, (_, remembered) in enumerate(MEMORY_STATUSES):
                    cell_mask = category_mask & (hits[:, :, simulation] == remembered)
                    values = lpp[:, :, simulation][cell_mask]
                    if values.size == 0:
                        raise ValueError(
                            f"Empty simulated cell in {model_id}: simulation "
                            f"{simulation}, participant {participant_id}"
                        )
                    participant_means[
                        participant_index, category_index, memory_index
                    ] = float(np.mean(values))
        simulation_means[simulation] = np.mean(participant_means, axis=0)

    if not np.isfinite(simulation_means).all():
        raise ValueError(f"Non-finite simulated summaries for {model_id}")
    return simulation_means, nonimputed_positions


def summarize(
    empirical_means: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray], int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        empirical_means.shape[0],
        size=(BOOTSTRAP_SAMPLES, empirical_means.shape[0]),
    )
    bootstrap_means = np.mean(empirical_means[indices], axis=1)
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    observed_mean = np.mean(empirical_means, axis=0)
    observed_bounds = np.quantile(bootstrap_means, [alpha, 1 - alpha], axis=0)

    arrays: dict[str, np.ndarray] = {"Observed": observed_mean}
    bounds: dict[str, np.ndarray] = {"Observed": observed_bounds}
    replicates: dict[str, np.ndarray] = {"Observed": bootstrap_means}
    nonimputed_positions = 0
    for model_id in ALL_MODEL_IDS:
        model_means, model_nonimputed_positions = load_simulation_summary(model_id)
        if nonimputed_positions not in {0, model_nonimputed_positions}:
            raise ValueError("Models disagree about the count of non-imputed inputs")
        nonimputed_positions = model_nonimputed_positions
        arrays[model_id] = np.mean(model_means, axis=0)
        bounds[model_id] = np.quantile(model_means, [alpha, 1 - alpha], axis=0)
        replicates[model_id] = model_means

    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for source_id in ("Observed", *ALL_MODEL_IDS):
        source_type = "observed" if source_id == "Observed" else "predicted"
        interval = (
            "Percentile bootstrap across participant cell means"
            if source_type == "observed"
            else "Central interval across complete simulated datasets"
        )
        n_units = EXPECTED_PARTICIPANTS if source_type == "observed" else EXPECTED_SIMULATIONS
        for category_index, (category, _) in enumerate(CATEGORIES):
            for memory_index, (status, _) in enumerate(MEMORY_STATUSES):
                summary_rows.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "source_label": SOURCE_LABELS[source_id],
                        "metric": "original_z_early_lpp",
                        "category": category,
                        "recall_status": status,
                        "mean": float(arrays[source_id][category_index, memory_index]),
                        "lower": float(bounds[source_id][0, category_index, memory_index]),
                        "upper": float(bounds[source_id][1, category_index, memory_index]),
                        "interval": interval,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "n_units": n_units,
                    }
                )
            contrast_rows.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_label": SOURCE_LABELS[source_id],
                    "contrast": f"Remembered minus Forgotten Early LPP: {category}",
                    "estimate": float(
                        arrays[source_id][category_index, 0]
                        - arrays[source_id][category_index, 1]
                    ),
                    "lower": float(
                        np.quantile(
                            replicates[source_id][:, category_index, 0]
                            - replicates[source_id][:, category_index, 1],
                            alpha,
                        )
                    ),
                    "upper": float(
                        np.quantile(
                            replicates[source_id][:, category_index, 0]
                            - replicates[source_id][:, category_index, 1],
                            1 - alpha,
                        )
                    ),
                    "interval": interval,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "n_units": n_units,
                }
            )

    if len(summary_rows) != 28 or len(contrast_rows) != 14:
        raise ValueError("Original-scale Early-LPP summary is incomplete")
    return summary_rows, contrast_rows, arrays, nonimputed_positions


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


def plot(summary_rows: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["source_id"]), str(row["category"]), str(row["recall_status"])): row
        for row in summary_rows
    }
    expected = {
        (source.source_id, category, status)
        for source in SOURCES
        for category, _ in CATEGORIES
        for status, _ in MEMORY_STATUSES
    }
    if not expected.issubset(lookup):
        raise ValueError(f"Missing selected Early-LPP cells: {sorted(expected - set(lookup))}")

    selected_rows = [lookup[key] for key in sorted(expected)]
    value_min = min(float(row["lower"]) for row in selected_rows)
    value_max = max(float(row["upper"]) for row in selected_rows)
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
        for panel_index, (axis, source) in enumerate(zip(axes, SOURCES, strict=True)):
            source_rows = [
                lookup[(source.source_id, category, status)]
                for category, _ in CATEGORIES
                for status, _ in MEMORY_STATUSES
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
                error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.05, "capsize": 2.7},
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
                expected_difference = float(
                    source_rows[start]["mean"]
                ) - float(source_rows[start + 1]["mean"])
                if not np.isclose(difference, expected_difference, rtol=0, atol=1e-12):
                    raise AssertionError(f"Delta mismatch for {source.source_id}")
                add_bracket(
                    axis,
                    POSITIONS[start],
                    POSITIONS[start + 1],
                    float(max(upper[start], upper[start + 1]) + bracket_gap),
                    bracket_tick,
                    difference,
                )

        figure.supylabel("Early LPP amplitude (z)", x=0.025, fontsize=10.5)
        figure.subplots_adjust(
            left=0.095,
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
    empirical_means, participant_rows, source_metadata = (
        load_empirical_participant_means()
    )
    summary_rows, contrast_rows, _, nonimputed_positions = summarize(empirical_means)

    _write_csv(
        PARTICIPANT_MEANS_PATH,
        participant_rows,
        ("participant", "category", "recall_status", "mean_early_lpp_z", "n_trials"),
    )
    _write_csv(
        SUMMARY_PATH,
        summary_rows,
        (
            "source_type",
            "source_id",
            "source_label",
            "metric",
            "category",
            "recall_status",
            "mean",
            "lower",
            "upper",
            "interval",
            "confidence_level",
            "n_units",
        ),
    )
    _write_csv(
        CONTRAST_PATH,
        contrast_rows,
        (
            "source_type",
            "source_id",
            "source_label",
            "contrast",
            "estimate",
            "lower",
            "upper",
            "interval",
            "confidence_level",
            "n_units",
        ),
    )
    source_metadata.update(
        {
            "observed_summary": (
                "Mean EarlyLPP within Participant × Condition × Memory, then "
                "mean across participants, matching the source R figure"
            ),
            "observed_interval": (
                f"{CONFIDENCE_LEVEL:.0%} percentile interval from "
                f"{BOOTSTRAP_SAMPLES:,} participant bootstrap samples"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "prediction_summary": (
                "Within each complete simulated dataset: mean original-z EarlyLPP "
                "within Participant × Condition × simulated recall status, then "
                "mean across participants"
            ),
            "prediction_interval": (
                f"Central {CONFIDENCE_LEVEL:.0%} interval across "
                f"{EXPECTED_SIMULATIONS} complete simulated datasets"
            ),
            "contrast_interval": (
                "Remembered-minus-Forgotten differences are calculated within "
                "each participant-bootstrap sample or complete simulated "
                "dataset before percentile limits are taken"
            ),
            "model_prediction_positions": nonimputed_positions,
            "model_prediction_exclusion": (
                "lpp_imputed == 1 positions are excluded because the empirical "
                "figure summarizes observed EEG values, not imputed fitting inputs"
            ),
            "comparison_note": (
                "The source empirical figure includes 39 EEG rows whose "
                "PresentationOrder is 99. Those rows contribute to the observed "
                "panel, as in the source R code, but cannot be assigned simulated "
                "recall status and therefore are absent from prediction summaries."
            ),
        }
    )
    SOURCE_PATH.write_text(json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8")
    plot(summary_rows)

    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {CONTRAST_PATH}")
    print(f"Saved {PARTICIPANT_MEANS_PATH}")
    print(f"Saved {SOURCE_PATH}")


if __name__ == "__main__":
    main()
