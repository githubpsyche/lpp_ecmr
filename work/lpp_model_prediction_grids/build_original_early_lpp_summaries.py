"""Build authenticated empirical and simulated original-z Early-LPP summaries.

The observed summaries follow the participant-level summarization in Robin
Hellerstedt's mixed-list R analysis and reads its original z-transformed
single-trial Early-LPP data from either the downloaded archive or its retained
extracted directory. Predicted summaries use the returned simulations but
summarize the uncentered ``EarlyLPP`` input rather than the within-list-centered
predictor used during fitting.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from lpp_ecmr.data_contract import (
    MIXED_EXPECTED_LISTS,
    MIXED_EXPECTED_SUBJECTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
FIT_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
DOWNLOAD_ARCHIVE = (
    PROJECT_ROOT.parent
    / "downloads"
    / "LPP_Zarubin_PureLists_for_Deborah.zip"
)
EXTRACTED_DELIVERY = DOWNLOAD_ARCHIVE.with_name(
    "LPP_Zarubin_PureLists_for_Deborah_extracted_2026-07-17"
)
EXTRACTED_DELIVERY = Path(
    os.environ.get("LPP_ECMR_EMPIRICAL_SOURCE_ROOT", EXTRACTED_DELIVERY)
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

BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
EXPECTED_PARTICIPANTS = MIXED_EXPECTED_SUBJECTS
EXPECTED_LISTS = MIXED_EXPECTED_LISTS
EXPECTED_SIMULATIONS = 200
EXPECTED_EMPIRICAL_ROWS = 6_508
EXPECTED_OUT_OF_RANGE_PRESENTATION_ROWS = 39

NEGATIVE = 1
NEUTRAL = 2
CATEGORIES = (("Negative", NEGATIVE), ("Neutral", NEUTRAL))
MEMORY_STATUSES = (("Remembered", True), ("Forgotten", False))

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

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _load_recorded_source(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing empirical-source provenance: {path}. The recorded member "
            "hashes are required before the source data can be used."
        )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected_members = {
        "data_member": EMPIRICAL_DATA_MEMBER,
        "analysis_member": EMPIRICAL_CODE_MEMBER,
    }
    for key, expected in expected_members.items():
        if recorded.get(key) != expected:
            raise ValueError(
                f"{path} records an unexpected {key}: {recorded.get(key)!r}"
            )
        hash_key = f"{key}_sha256"
        digest = recorded.get(hash_key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{path} has no valid {hash_key}")
    return recorded


def _load_empirical_source_bytes(
    *,
    archive_path: Path = DOWNLOAD_ARCHIVE,
    extracted_root: Path = EXTRACTED_DELIVERY,
    provenance_path: Path = SOURCE_PATH,
) -> tuple[bytes, bytes, dict[str, object]]:
    """Load and authenticate the two retained empirical source members."""

    recorded = _load_recorded_source(provenance_path)
    archive_path = Path(archive_path)
    extracted_root = Path(extracted_root)

    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            required = {EMPIRICAL_DATA_MEMBER, EMPIRICAL_CODE_MEMBER}
            if not required.issubset(names):
                raise FileNotFoundError(
                    "Empirical archive is missing members: "
                    f"{sorted(required - names)}"
                )
            empirical_bytes = archive.read(EMPIRICAL_DATA_MEMBER)
            analysis_bytes = archive.read(EMPIRICAL_CODE_MEMBER)
        source_metadata: dict[str, object] = {
            "source_type": "zip_archive",
            "source_path": str(archive_path),
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
        }
    else:
        empirical_path = extracted_root / EMPIRICAL_DATA_MEMBER
        analysis_path = extracted_root / EMPIRICAL_CODE_MEMBER
        missing = [
            str(path)
            for path in (empirical_path, analysis_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Neither the empirical archive nor a complete extracted delivery "
                f"is available. Missing extracted members: {missing}"
            )
        empirical_bytes = empirical_path.read_bytes()
        analysis_bytes = analysis_path.read_bytes()
        source_metadata = {
            "source_type": "extracted_directory",
            "source_path": str(extracted_root),
            # Retain the original delivery identity even though the redundant ZIP
            # no longer needs to remain on disk.
            "archive": recorded.get("archive"),
            "archive_sha256": recorded.get("archive_sha256"),
        }

    member_hashes = {
        "data_member_sha256": _bytes_sha256(empirical_bytes),
        "analysis_member_sha256": _bytes_sha256(analysis_bytes),
    }
    for key, actual in member_hashes.items():
        expected = recorded[key]
        if actual != expected:
            raise ValueError(
                f"Empirical source {key} mismatch: expected {expected}, found {actual}"
            )

    source_metadata.update(
        {
            "source_provenance": str(provenance_path),
            "data_member": EMPIRICAL_DATA_MEMBER,
            "data_member_sha256": member_hashes["data_member_sha256"],
            "analysis_member": EMPIRICAL_CODE_MEMBER,
            "analysis_member_sha256": member_hashes["analysis_member_sha256"],
        }
    )
    return empirical_bytes, analysis_bytes, source_metadata


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_empirical_participant_means() -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Reproduce the R analysis's participant-by-cell Early-LPP means."""
    empirical_bytes, analysis_bytes, source_metadata = (
        _load_empirical_source_bytes()
    )

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

    metadata = source_metadata | {
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
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {CONTRAST_PATH}")
    print(f"Saved {PARTICIPANT_MEANS_PATH}")
    print(f"Saved {SOURCE_PATH}")


if __name__ == "__main__":
    main()
