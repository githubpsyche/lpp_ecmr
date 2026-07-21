"""Build the combined mixed- and pure-list Talmi EEG HDF5 dataset.

The builder preserves the established 342 mixed rows exactly and appends 297
pure-list rows. Reproducibility metadata is returned in memory and may be
written to JSON when explicitly requested; correctness does not depend on a
sidecar. Nothing is written unless :func:`write_build` or the command-line
interface is called with an explicit output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from lpp_ecmr.data_contract import (
    COND3_CODES,
    CONDITION_CODES,
    LIST_TYPE_CODES,
    LIST_TYPE_MIXED,
    LIST_TYPE_PURE_NEGATIVE,
    LIST_TYPE_PURE_NEUTRAL,
    MIXED_TRIAL_QUERY,
    TARGET_LIST_LENGTH,
    DataContractError,
    validate_combined_dataset,
    validate_legacy_mixed_semantics,
)

PURE_SUBJECT_IDS = (
    101,
    102,
    104,
    105,
    106,
    107,
    110,
    111,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    133,
    135,
    136,
    140,
    141,
    142,
)

MIXED_BEHAVIOR_COLUMNS = (
    "Participant",
    "List",
    "TrialNumber",
    "ThreeCondsCode",
    "ConditionCode",
    "ThreeConds",
    "Recalled",
)
EEG_COLUMNS = (
    "Participant",
    "List",
    "EEGThreeConds",
    "EEGCondition",
    "TrialNumber",
    "EEGPresentationOrder",
    "EEGImgName",
    "EEGRecalled",
    "EarlyLPP",
    "LateLPP",
)
PURE_BEHAVIOR_COLUMNS = (
    "List",
    "TrialNumber",
    "ThreeConds",
    "RawRecall",
    "ImgName",
)
NUMERIC_EEG_COLUMNS = (
    "Participant",
    "List",
    "TrialNumber",
    "EEGPresentationOrder",
    "EEGRecalled",
    "EarlyLPP",
    "LateLPP",
)


@dataclass(frozen=True)
class BuildInputs:
    """Explicit source paths for a reproducible combined build."""

    mixed_behavior: Path
    mixed_eeg: Path
    pure_behavior_dir: Path
    pure_eeg: Path


@dataclass
class BuildResult:
    """In-memory combined data plus metadata ready for writing."""

    dataset: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _read_fixed_width_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != len(columns):
        raise DataContractError(
            f"{path} has {frame.shape[1]} columns; expected {len(columns)}"
        )
    frame.columns = list(columns)
    if str(frame.iloc[0, 0]).strip().lower() in {
        columns[0].lower(),
        "participant",
        "list",
    }:
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


def _coerce_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: Path,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            rows = result.index[result[column].isna()].tolist()[:5]
            raise DataContractError(
                f"{source}: non-numeric or missing {column} at rows {rows}"
            )
    return result


def _strip_image_suffix(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)


def load_mixed_behavior(path: Path) -> pd.DataFrame:
    """Load the mixed behavioral master and retain its canonical 20 rows/list."""

    frame = _read_fixed_width_csv(Path(path), MIXED_BEHAVIOR_COLUMNS)
    numeric = (
        "Participant",
        "List",
        "TrialNumber",
        "ThreeCondsCode",
        "ConditionCode",
        "Recalled",
    )
    frame = _coerce_numeric(frame, numeric, source=Path(path))
    frame["PresentationOrder"] = frame["TrialNumber"] - 2
    frame = frame[frame["PresentationOrder"].between(1, TARGET_LIST_LENGTH)].copy()
    frame["Condition"] = frame["ConditionCode"].map({1: "Negative", 2: "Neutral"})
    if frame["Condition"].isna().any():
        raise DataContractError("Mixed behavior contains an unknown ConditionCode")
    frame["Recalled"] = frame["Recalled"].astype(np.int64)
    frame["Participant"] = frame["Participant"].astype(np.int64)
    frame["List"] = frame["List"].astype(np.int64)
    frame["PresentationOrder"] = frame["PresentationOrder"].astype(np.int64)
    frame["list_type"] = LIST_TYPE_MIXED
    return frame


def load_eeg_csv(path: Path) -> pd.DataFrame:
    """Load either Robin EEG CSV, accepting headered or headerless files."""

    frame = _read_fixed_width_csv(Path(path), EEG_COLUMNS)
    frame = _coerce_numeric(frame, NUMERIC_EEG_COLUMNS, source=Path(path))
    for column in ("Participant", "List", "TrialNumber", "EEGPresentationOrder"):
        frame[column] = frame[column].astype(np.int64)
    frame["EEGImgName"] = _strip_image_suffix(frame["EEGImgName"])
    frame["EEGThreeConds"] = frame["EEGThreeConds"].astype(str).str.strip()
    frame["EEGCondition"] = frame["EEGCondition"].astype(str).str.strip()
    return frame


def prepare_mixed_dataframe(
    behavior_path: Path,
    eeg_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the established mixed event table without changing semantics."""

    behavior = load_mixed_behavior(behavior_path)
    eeg_all = load_eeg_csv(eeg_path)
    eeg = eeg_all[eeg_all["EEGPresentationOrder"].between(1, TARGET_LIST_LENGTH)].copy()
    join_keys = ["Participant", "List", "PresentationOrder"]
    eeg = eeg.rename(columns={"EEGPresentationOrder": "PresentationOrder"})
    if eeg.duplicated(join_keys).any():
        raise DataContractError(
            "Mixed EEG has duplicate participant/list/position rows"
        )

    merged = behavior.merge(
        eeg[
            [
                "Participant",
                "List",
                "PresentationOrder",
                "EEGImgName",
                "EarlyLPP",
                "LateLPP",
            ]
        ],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    missing_image = merged["EEGImgName"].isna()
    merged["ImgName"] = merged["EEGImgName"]
    merged.loc[missing_image, "ImgName"] = merged.loc[missing_image].apply(
        lambda row: (
            f"missing_{int(row.Participant)}_{int(row.List)}_"
            f"{int(row.PresentationOrder)}"
        ),
        axis=1,
    )
    merged["imputed_mask"] = missing_image.astype(np.int64)
    merged = impute_lpp_within_cohort(merged)
    _validate_event_grid(merged, cohort="mixed")
    diagnostics = {
        "behavior_events": int(len(behavior)),
        "eeg_rows_total": int(len(eeg_all)),
        "eeg_rows_used": int(len(eeg)),
        "eeg_rows_source_position_99": int(
            (eeg_all["EEGPresentationOrder"] == 99).sum()
        ),
        "lpp_missing_events": int(merged["lpp_imputed"].sum()),
    }
    return merged, diagnostics


def load_pure_behavior_logs(
    directory: Path,
    *,
    subject_ids: Sequence[int] = PURE_SUBJECT_IDS,
    expected_lists: int = 9,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load included pure behavioral logs and derive positions after buffers."""

    directory = Path(directory)
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for subject in subject_ids:
        path = directory / f"HT_{subject}.csv"
        paths.append(path)
        frame = _read_fixed_width_csv(path, PURE_BEHAVIOR_COLUMNS)
        frame = _coerce_numeric(
            frame,
            ("List", "TrialNumber", "RawRecall"),
            source=path,
        )
        frame.insert(0, "Participant", int(subject))
        frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)

    behavior_all = pd.concat(frames, ignore_index=True)
    behavior_all["ThreeConds"] = behavior_all["ThreeConds"].astype(str).str.strip()
    behavior_all["ImgName"] = _strip_image_suffix(behavior_all["ImgName"])

    per_subject_rows = behavior_all.groupby("Participant").size()
    expected_rows = expected_lists * (TARGET_LIST_LENGTH + 2)
    if not np.all(per_subject_rows == expected_rows):
        raise DataContractError(
            f"Pure behavior must have {expected_rows} rows per subject"
        )
    per_list_rows = behavior_all.groupby(["Participant", "List"]).size()
    if not np.all(per_list_rows == TARGET_LIST_LENGTH + 2):
        raise DataContractError(
            "Every pure list must contain 22 rows including buffers"
        )

    buffer_counts = (
        behavior_all.assign(
            _is_buffer=behavior_all["ThreeConds"].eq("Buffer").astype(int)
        )
        .groupby(["Participant", "List"])["_is_buffer"]
        .sum()
    )
    if not np.all(buffer_counts == 2):
        raise DataContractError("Every pure list must contain exactly two Buffer rows")

    behavior = behavior_all[behavior_all["ThreeConds"] != "Buffer"].copy()
    behavior["PresentationOrder"] = (
        behavior.groupby(["Participant", "List"], sort=False).cumcount() + 1
    )
    behavior["Condition"] = behavior["ThreeConds"].map(
        {"Negative": "Negative", "Neutral": "Neutral", "Category": "Neutral"}
    )
    if behavior["Condition"].isna().any():
        unknown = sorted(
            behavior.loc[behavior["Condition"].isna(), "ThreeConds"].unique()
        )
        raise DataContractError(f"Unknown pure ThreeConds values: {unknown}")

    behavior["Recalled"] = behavior["RawRecall"].eq(1).astype(np.int64)
    behavior["list_type"] = behavior["ThreeConds"].map(
        {
            "Negative": LIST_TYPE_PURE_NEGATIVE,
            "Neutral": LIST_TYPE_PURE_NEUTRAL,
            "Category": LIST_TYPE_PURE_NEUTRAL,
        }
    )
    for column in ("Participant", "List", "TrialNumber", "PresentationOrder"):
        behavior[column] = behavior[column].astype(np.int64)

    condition_counts = behavior.groupby(["Participant", "List"])["ThreeConds"].nunique()
    if not np.all(condition_counts == 1):
        raise DataContractError("Every pure list must be homogeneous in ThreeConds")
    _validate_event_grid(behavior, cohort="pure behavior")
    return behavior, paths


def prepare_pure_dataframe(
    behavior_dir: Path,
    eeg_path: Path,
    *,
    subject_ids: Sequence[int] = PURE_SUBJECT_IDS,
    expected_lists: int = 9,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Complete pure lists from behavior and align the available EEG rows."""

    behavior, behavior_paths = load_pure_behavior_logs(
        behavior_dir,
        subject_ids=subject_ids,
        expected_lists=expected_lists,
    )
    eeg = load_eeg_csv(eeg_path)
    join_keys = ["Participant", "List", "TrialNumber"]
    if eeg.duplicated(join_keys).any():
        raise DataContractError("Pure EEG has duplicate participant/list/trial rows")

    key_check = eeg[join_keys].merge(
        behavior[join_keys],
        on=join_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if (key_check["_merge"] != "both").any():
        raise DataContractError("Pure EEG contains rows absent from behavioral logs")

    merged = behavior.merge(
        eeg[
            join_keys
            + [
                "EEGPresentationOrder",
                "EEGThreeConds",
                "EEGCondition",
                "EEGImgName",
                "EEGRecalled",
                "EarlyLPP",
                "LateLPP",
            ]
        ],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    eeg_present = merged["EEGImgName"].notna()
    image_mismatch = eeg_present & merged["ImgName"].ne(merged["EEGImgName"])
    if image_mismatch.any():
        rows = merged.index[image_mismatch].tolist()[:5]
        raise DataContractError(f"Pure behavior/EEG image mismatch at rows {rows}")
    condition_mismatch = eeg_present & merged["ThreeConds"].ne(merged["EEGThreeConds"])
    if condition_mismatch.any():
        rows = merged.index[condition_mismatch].tolist()[:5]
        raise DataContractError(f"Pure behavior/EEG condition mismatch at rows {rows}")

    expected_eeg_recalled = merged["Recalled"].astype(float)
    recall_mismatch = (
        eeg_present
        & merged["EEGRecalled"].notna()
        & merged["EEGRecalled"].ne(expected_eeg_recalled)
    )
    if recall_mismatch.any():
        rows = merged.index[recall_mismatch].tolist()[:5]
        raise DataContractError(f"Pure behavior/EEG recall mismatch at rows {rows}")

    merged["imputed_mask"] = (~eeg_present).astype(np.int64)
    merged = impute_lpp_within_cohort(merged)
    _validate_event_grid(merged, cohort="pure")

    per_list_missing = merged.groupby(["Participant", "List"])["lpp_imputed"].sum()
    diagnostics = {
        "behavior_events": int(len(behavior)),
        "eeg_rows_total": int(len(eeg)),
        "eeg_rows_source_position_99": int((eeg["EEGPresentationOrder"] == 99).sum()),
        "lpp_missing_events": int(merged["lpp_imputed"].sum()),
        "lists_with_missing_lpp": int((per_list_missing > 0).sum()),
        "fully_missing_lpp_lists": [
            {"participant": int(participant), "list": int(list_id)}
            for participant, list_id in per_list_missing[
                per_list_missing == TARGET_LIST_LENGTH
            ].index
        ],
        "raw_recall_counts": {
            str(value): int(count)
            for value, count in merged["RawRecall"].value_counts().sort_index().items()
        },
    }
    return merged, diagnostics, behavior_paths


def impute_lpp_within_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing LPP without borrowing information across list cohorts."""

    result = frame.copy()
    lpp_columns = ["EarlyLPP", "LateLPP"]
    initial_missing = result[lpp_columns].isna().any(axis=1)
    condition_means = result.groupby(["Participant", "List", "Condition"])[
        lpp_columns
    ].transform("mean")
    list_means = result.groupby(["Participant", "List"])[lpp_columns].transform("mean")
    cohort_means = result.groupby("Condition")[lpp_columns].mean()

    for column in lpp_columns:
        result[column] = result[column].fillna(condition_means[column])
        result[column] = result[column].fillna(list_means[column])
        result[column] = result[column].fillna(
            result["Condition"].map(cohort_means[column])
        )
        result[column] = result[column].fillna(0.0)
    result["lpp_imputed"] = initial_missing.astype(np.int64)
    return result


def _validate_event_grid(frame: pd.DataFrame, *, cohort: str) -> None:
    groups = frame.groupby(["Participant", "List"], sort=True)
    counts = groups.size()
    if not np.all(counts == TARGET_LIST_LENGTH):
        bad = counts[counts != TARGET_LIST_LENGTH].head().to_dict()
        raise DataContractError(f"{cohort} has incomplete lists: {bad}")
    for key, block in groups:
        positions = np.sort(block["PresentationOrder"].to_numpy(dtype=int))
        if not np.array_equal(positions, np.arange(1, TARGET_LIST_LENGTH + 1)):
            raise DataContractError(
                f"{cohort} list {key} does not contain positions 1..20"
            )


def assign_shared_item_codes(
    mixed: pd.DataFrame,
    pure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assign one sorted item namespace while preserving established mixed codes."""

    combined_names = pd.concat(
        [mixed["ImgName"], pure["ImgName"]],
        ignore_index=True,
    )
    codes, categories = pd.factorize(combined_names, sort=True)
    mixed_result = mixed.copy()
    pure_result = pure.copy()
    mixed_result["item_id_code"] = codes[: len(mixed)] + 1
    pure_result["item_id_code"] = codes[len(mixed) :] + 1
    return mixed_result, pure_result, [str(value) for value in categories]


def assign_condition_codes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["condition"] = result["Condition"].map(CONDITION_CODES)
    result["cond3"] = result["ThreeConds"].map(COND3_CODES)
    if result[["condition", "cond3"]].isna().any().any():
        raise DataContractError("Failed to encode condition or cond3")
    result["condition"] = result["condition"].astype(np.int64)
    result["cond3"] = result["cond3"].astype(np.int64)
    return result


def build_recall_dataset(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a completed event table into trial-major RecallDataset arrays."""

    _validate_event_grid(frame, cohort="dataset")
    ordered = frame.sort_values(["Participant", "List", "PresentationOrder"])
    groups = list(ordered.groupby(["Participant", "List"], sort=True))
    trial_count = len(groups)

    subject = np.zeros((trial_count, 1), dtype=np.int64)
    list_ids = np.zeros((trial_count, 1), dtype=np.int64)
    list_length = np.full(
        (trial_count, 1),
        TARGET_LIST_LENGTH,
        dtype=np.int64,
    )
    list_type = np.zeros((trial_count, 1), dtype=np.int64)
    pres_itemnos = np.tile(
        np.arange(1, TARGET_LIST_LENGTH + 1, dtype=np.int64),
        (trial_count, 1),
    )
    pres_itemids = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.int64)
    early_lpp = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.float64)
    late_lpp = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.float64)
    condition = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.int64)
    cond3 = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.int64)
    lpp_imputed = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.int64)
    recalls = np.zeros((trial_count, TARGET_LIST_LENGTH), dtype=np.int64)
    restricted_recalls = np.zeros_like(recalls)
    rec_itemids = np.zeros_like(recalls)

    for trial_index, ((participant, list_id), block) in enumerate(groups):
        block = block.sort_values("PresentationOrder").reset_index(drop=True)
        block_list_types = block["list_type"].unique()
        if block_list_types.size != 1:
            raise DataContractError(
                f"Participant/list {(participant, list_id)} has mixed list_type codes"
            )
        subject[trial_index, 0] = int(participant)
        list_ids[trial_index, 0] = int(list_id)
        list_type[trial_index, 0] = int(block_list_types[0])
        pres_itemids[trial_index] = block["item_id_code"].to_numpy(dtype=np.int64)
        early_lpp[trial_index] = block["EarlyLPP"].to_numpy(dtype=np.float64)
        late_lpp[trial_index] = block["LateLPP"].to_numpy(dtype=np.float64)
        condition[trial_index] = block["condition"].to_numpy(dtype=np.int64)
        cond3[trial_index] = block["cond3"].to_numpy(dtype=np.int64)
        lpp_imputed[trial_index] = block["lpp_imputed"].to_numpy(dtype=np.int64)

        recalled = block["Recalled"].eq(1)
        recalled_positions = block.loc[recalled, "PresentationOrder"].to_numpy(
            dtype=np.int64
        )
        recalled_itemids = block.loc[recalled, "item_id_code"].to_numpy(dtype=np.int64)
        recall_count = recalled_positions.size
        recalls[trial_index, :recall_count] = recalled_positions
        restricted_recalls[trial_index, :recall_count] = recalled_positions
        rec_itemids[trial_index, :recall_count] = recalled_itemids

    return {
        "subject": subject,
        "list": list_ids,
        "listLength": list_length,
        "list_type": list_type,
        "pres_itemnos": pres_itemnos,
        "pres_itemids": pres_itemids,
        "recalls": recalls,
        "restricted_recalls": restricted_recalls,
        "rec_itemids": rec_itemids,
        "EarlyLPP": early_lpp,
        "LateLPP": late_lpp,
        "condition": condition,
        "cond3": cond3,
        "lpp_imputed": lpp_imputed,
    }


def concatenate_datasets(
    datasets: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not datasets:
        raise DataContractError("No datasets supplied for concatenation")
    expected_keys = set(datasets[0])
    for dataset in datasets[1:]:
        if set(dataset) != expected_keys:
            raise DataContractError("Datasets do not have identical fields")
    return {
        key: np.concatenate([np.asarray(dataset[key]) for dataset in datasets], axis=0)
        for key in datasets[0]
    }


def read_hdf5_dataset(path: Path) -> dict[str, np.ndarray]:
    """Read the project HDF5 representation without JAX dtype conversion."""

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise DataContractError(f"{path} does not contain a /data group")
        return {key: np.asarray(value[()]).T for key, value in handle["data"].items()}


def assert_mixed_compatibility(
    combined: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
) -> None:
    """Require every established field to match the old mixed H5 exactly."""

    list_types = np.asarray(combined["list_type"]).reshape(-1)
    mixed_mask = list_types == LIST_TYPE_MIXED
    for key, baseline_value in baseline.items():
        if key == "list_type":
            continue
        if key not in combined:
            raise DataContractError(f"Combined dataset lacks baseline field {key}")
        actual = np.asarray(combined[key])[mixed_mask]
        expected = np.asarray(baseline_value)
        if actual.shape != expected.shape:
            raise DataContractError(
                f"Mixed compatibility shape mismatch for {key}: "
                f"{actual.shape} != {expected.shape}"
            )
        if not np.array_equal(actual, expected):
            if np.issubdtype(actual.dtype, np.floating):
                max_difference = float(np.max(np.abs(actual - expected)))
                detail = f"; max absolute difference {max_difference}"
            else:
                detail = (
                    f"; differing cells {int(np.count_nonzero(actual != expected))}"
                )
            raise DataContractError(f"Mixed compatibility failed for {key}{detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_combined_dataset(
    inputs: BuildInputs,
    *,
    baseline_h5: Path | None = None,
) -> BuildResult:
    """Build and validate the approved 639-trial combined dataset."""

    mixed, mixed_diagnostics = prepare_mixed_dataframe(
        inputs.mixed_behavior,
        inputs.mixed_eeg,
    )
    pure, pure_diagnostics, pure_behavior_paths = prepare_pure_dataframe(
        inputs.pure_behavior_dir,
        inputs.pure_eeg,
    )
    mixed, pure, item_categories = assign_shared_item_codes(mixed, pure)
    mixed = assign_condition_codes(mixed)
    pure = assign_condition_codes(pure)

    mixed_dataset = build_recall_dataset(mixed)
    pure_dataset = build_recall_dataset(pure)
    combined = concatenate_datasets([mixed_dataset, pure_dataset])
    summary = validate_combined_dataset(combined)
    mixed_semantic_digests = validate_legacy_mixed_semantics(combined)

    compatibility = {"checked": False}
    if baseline_h5 is not None:
        baseline = read_hdf5_dataset(Path(baseline_h5))
        assert_mixed_compatibility(combined, baseline)
        compatibility = {
            "checked": True,
            "baseline": _source_record(Path(baseline_h5)),
            "matching_fields": sorted(baseline),
        }

    metadata = {
        "schema_version": 1,
        "list_type_codes": LIST_TYPE_CODES,
        "mixed_trial_query": MIXED_TRIAL_QUERY,
        "condition_codes": CONDITION_CODES,
        "cond3_codes": COND3_CODES,
        "recall_scoring": {
            "rule": "strict equality: RawRecall == 1",
            "nonrecalled_source_values": [0, 0.25, 0.5, 0.75, 2],
            "restricted_recalls": "identical to recalls",
        },
        "lpp_imputation": [
            "within participant/list/collapsed condition",
            "within participant/list",
            "within cohort/collapsed condition",
            "zero fallback",
        ],
        "summary": summary,
        "diagnostics": {
            "mixed": mixed_diagnostics,
            "pure": pure_diagnostics,
        },
        "item_categories": item_categories,
        "sources": {
            "mixed_behavior": _source_record(inputs.mixed_behavior),
            "mixed_eeg": _source_record(inputs.mixed_eeg),
            "pure_eeg": _source_record(inputs.pure_eeg),
            "pure_behavior": [_source_record(path) for path in pure_behavior_paths],
        },
        "mixed_compatibility": compatibility,
        "legacy_mixed_semantic_digests": mixed_semantic_digests,
    }
    return BuildResult(dataset=combined, metadata=metadata)


def write_hdf5_dataset(dataset: Mapping[str, np.ndarray], path: Path) -> None:
    """Write trial-major arrays in the transposed project HDF5 representation."""

    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        for key, value in dataset.items():
            data_group.create_dataset(key, data=np.asarray(value).T)


def write_build(
    result: BuildResult,
    output: Path,
    *,
    metadata_output: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path | None]:
    """Write the H5 and, only when requested, an optional metadata sidecar."""

    output = Path(output)
    paths = [output]
    if metadata_output is not None:
        metadata_output = Path(metadata_output)
        paths.append(metadata_output)
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    write_hdf5_dataset(result.dataset, output)
    if metadata_output is not None:
        metadata = dict(result.metadata)
        metadata["output"] = _source_record(output)
        metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output, metadata_output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the combined 639-trial Talmi EEG HDF5 dataset."
    )
    parser.add_argument("--mixed-behavior", required=True, type=Path)
    parser.add_argument("--mixed-eeg", required=True, type=Path)
    parser.add_argument("--pure-behavior-dir", required=True, type=Path)
    parser.add_argument("--pure-eeg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--baseline-h5",
        type=Path,
        help="Optional established mixed H5 to require exact compatibility against.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output files if they already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = BuildInputs(
        mixed_behavior=args.mixed_behavior,
        mixed_eeg=args.mixed_eeg,
        pure_behavior_dir=args.pure_behavior_dir,
        pure_eeg=args.pure_eeg,
    )
    result = build_combined_dataset(inputs, baseline_h5=args.baseline_h5)
    output, metadata = write_build(
        result,
        args.output,
        metadata_output=args.metadata_output,
        force=args.force,
    )
    summary = result.metadata["summary"]
    message = f"Wrote {summary['trial_count']} trials to {output}"
    if metadata is not None:
        message += f" and metadata to {metadata}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
