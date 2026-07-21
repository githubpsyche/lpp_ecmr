"""Schema and validation rules for the combined Talmi EEG recall dataset.

The public contract is deliberately small: trial-major arrays, twenty study
positions per trial, and a three-level ``list_type`` field.  HDF5 storage is
handled elsewhere and transposes these arrays for compatibility with
``jaxcmr.helpers.load_data``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

TARGET_LIST_LENGTH = 20

LIST_TYPE_MIXED = 0
LIST_TYPE_PURE_NEGATIVE = 1
LIST_TYPE_PURE_NEUTRAL = 2
LIST_TYPE_CODES = {
    "mixed": LIST_TYPE_MIXED,
    "pure_negative": LIST_TYPE_PURE_NEGATIVE,
    "pure_neutral": LIST_TYPE_PURE_NEUTRAL,
}
MIXED_TRIAL_QUERY = "data['list_type'] == 0"

CONDITION_CODES = {"Negative": 1, "Neutral": 2}
COND3_CODES = {"Negative": 1, "Neutral": 2, "Category": 3}

EXPECTED_MIXED_TRIALS = 342
EXPECTED_PURE_NEGATIVE_TRIALS = 99
EXPECTED_PURE_NEUTRAL_TRIALS = 198
EXPECTED_COMBINED_TRIALS = 639
EXPECTED_SUBJECTS = 71
EXPECTED_LISTS_PER_SUBJECT = 9
EXPECTED_ITEM_CODE_MAX = 569
EXPECTED_LPP_IMPUTED_ITEMS = 695
EXPECTED_RECALL_EVENTS = 5_780

# Consumer-facing aliases use the terminology already present in fitting
# registries: one trial corresponds to one studied list.
MIXED_EXPECTED_LISTS = EXPECTED_MIXED_TRIALS
MIXED_EXPECTED_SUBJECTS = 38
LEGACY_MIXED_H5_SHA256 = (
    "adea78c12ef9ce73de6e8bba95ce37d524aaeea3706780f39e4e0c95cc5981e3"
)

# SHA-256 digests of the loaded, trial-major arrays in the established
# mixed-only TalmiEEG.h5 (whole-file SHA-256 above).
# These are executable compatibility gates after that file is replaced.
LEGACY_MIXED_FIELD_DIGESTS = {
    "EarlyLPP": "10c0b38c9cdd0eb9d44c1cccede8ea751b560e548551e14e53d6503772a3c00e",
    "LateLPP": "d7847f0bbdbf3c5fc4cb24d4320d8d8e7f7cfd2dc2e7085cdfeebee59f34fb28",
    "cond3": "af72abad810def74200784b7580b5d3f5b5743d4f748ae51f1b45165930d0ad2",
    "condition": "1bdb8a663e1b49721d803040b3da9ab954222d79700432c927265eeaa4102d53",
    "list": "b8a01608a66620275b5b2d4d32247a65452778da4c64b361be1b2cf47191fd6a",
    "listLength": "7ada7f4c2393187e20609c63cd246104239b85c1085a5c9452530bad4230c0fd",
    "lpp_imputed": "366409f2c822f1191a9dec30ce694b3b6bf242823d332263b97a231c03bad57b",
    "pres_itemids": "c410bfb957943c828575090424708030ea377f38c8e20047e9ca54106d5bb978",
    "pres_itemnos": "8448b519635647d328715c723397528f595bc79e02084193758b4f95de4b18e9",
    "rec_itemids": "70447b5350781519c11f2ba5e9687d20fb98f13dcfff6a086f26485907168f1b",
    "recalls": "d743c50cc5a813b392eed253323ec53d61852c00fc57f7de1f3799002603b1d0",
    "restricted_recalls": (
        "d743c50cc5a813b392eed253323ec53d61852c00fc57f7de1f3799002603b1d0"
    ),
    "subject": "21b89d63880511bcf8c5232f49c29405345a9535e0a40c1725bc59be709847ce",
}

TRIAL_FIELDS = ("subject", "list", "listLength", "list_type")
POSITION_FIELDS = (
    "pres_itemnos",
    "pres_itemids",
    "EarlyLPP",
    "LateLPP",
    "condition",
    "cond3",
    "lpp_imputed",
)
RECALL_FIELDS = ("recalls", "restricted_recalls", "rec_itemids")
REQUIRED_FIELDS = TRIAL_FIELDS + POSITION_FIELDS + RECALL_FIELDS


class DataContractError(ValueError):
    """Raised when a dataset violates the combined Talmi EEG contract."""


def semantic_array_digest(value: Any) -> str:
    """Hash dtype, shape, and C-order bytes of one trial-major array."""

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_legacy_mixed_semantics(
    combined: Mapping[str, Any],
) -> dict[str, str]:
    """Require mixed rows to match durable legacy per-field digests.

    The input must retain the raw HDF5 storage dtypes (currently int64 and
    float64), as returned by ``data_preparation.read_hdf5_dataset`` or by the
    builder itself.  This deliberately does not accept the int32/float32
    conversion performed by ``jaxcmr.helpers.load_data``.
    """

    list_types = _array(combined, "list_type").reshape(-1)
    mixed_mask = list_types == LIST_TYPE_MIXED
    if int(mixed_mask.sum()) != EXPECTED_MIXED_TRIALS:
        raise DataContractError(
            f"Expected {EXPECTED_MIXED_TRIALS} mixed rows for digest validation"
        )

    actual: dict[str, str] = {}
    for key, expected_digest in LEGACY_MIXED_FIELD_DIGESTS.items():
        value = _array(combined, key)[mixed_mask]
        actual_digest = semantic_array_digest(value)
        actual[key] = actual_digest
        if actual_digest != expected_digest:
            raise DataContractError(
                f"Legacy mixed semantic digest failed for {key}: "
                f"{actual_digest} != {expected_digest}"
            )
    return actual


def mixed_trial_mask(
    dataset: Mapping[str, Any],
    *,
    allow_legacy_missing_field: bool = False,
) -> np.ndarray:
    """Return a one-dimensional mask selecting mixed trials.

    New data and regenerated simulations must carry ``list_type`` explicitly.
    The opt-in legacy mode exists only for migration tooling that compares an
    old artifact before replacing it.
    """

    if "subject" not in dataset:
        raise DataContractError("Cannot infer trial count without subject")
    subject = np.asarray(dataset["subject"])
    if subject.ndim == 0:
        raise DataContractError("subject must have a trial axis")
    trial_count = subject.shape[0]

    if "list_type" not in dataset:
        if not allow_legacy_missing_field:
            raise DataContractError("Dataset does not contain list_type")
        return np.ones(trial_count, dtype=bool)

    list_type = np.asarray(dataset["list_type"])
    if list_type.shape not in {(trial_count,), (trial_count, 1)}:
        raise DataContractError(
            "list_type must have shape "
            f"({trial_count},) or ({trial_count}, 1); found {list_type.shape}"
        )
    values = list_type.reshape(-1)
    unexpected = set(np.unique(values)) - set(LIST_TYPE_CODES.values())
    if unexpected:
        raise DataContractError(f"Unexpected list_type values: {sorted(unexpected)}")
    return values == LIST_TYPE_MIXED


def slice_trials(
    dataset: Mapping[str, Any],
    trial_mask: Any,
) -> dict[str, Any]:
    """Slice every trial-major dataset field with one validated boolean mask."""

    mask = np.asarray(trial_mask, dtype=bool)
    if mask.ndim != 1:
        raise DataContractError(
            f"trial_mask must be one-dimensional; found {mask.shape}"
        )
    trial_count = mask.size
    selected: dict[str, Any] = {}
    for key, value in dataset.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != trial_count:
            raise DataContractError(
                f"{key} is not aligned to the {trial_count}-trial mask; "
                f"found shape {array.shape}"
            )
        selected[key] = value[mask]
    return selected


def _array(dataset: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in dataset:
        raise DataContractError(f"Missing required dataset field: {key}")
    value = np.asarray(dataset[key])
    if value.ndim != 2:
        raise DataContractError(f"{key} must be two-dimensional; found {value.shape}")
    return value


def _require_integer(key: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.integer):
        raise DataContractError(
            f"{key} must have an integer dtype; found {value.dtype}"
        )


def _validate_recall_rows(dataset: Mapping[str, Any]) -> None:
    recalls = _array(dataset, "recalls")
    restricted = _array(dataset, "restricted_recalls")
    rec_itemids = _array(dataset, "rec_itemids")
    pres_itemids = _array(dataset, "pres_itemids")

    if not np.array_equal(recalls, restricted):
        raise DataContractError(
            "restricted_recalls must remain identical to recalls for this dataset"
        )

    for trial_index, (recall_row, item_row) in enumerate(
        zip(recalls, rec_itemids, strict=True)
    ):
        recall_count = int(np.count_nonzero(recall_row))
        item_count = int(np.count_nonzero(item_row))
        if recall_count != item_count:
            raise DataContractError(
                f"Trial {trial_index} has {recall_count} recalls but "
                f"{item_count} recalled item IDs"
            )
        if np.any(recall_row[recall_count:] != 0):
            raise DataContractError(
                f"Trial {trial_index} has non-trailing recall padding"
            )
        if np.any(item_row[item_count:] != 0):
            raise DataContractError(
                f"Trial {trial_index} has non-trailing recalled-item padding"
            )
        recalled_positions = recall_row[:recall_count]
        if np.any((recalled_positions < 1) | (recalled_positions > TARGET_LIST_LENGTH)):
            raise DataContractError(
                f"Trial {trial_index} has a recall outside 1..{TARGET_LIST_LENGTH}"
            )
        if np.unique(recalled_positions).size != recalled_positions.size:
            raise DataContractError(f"Trial {trial_index} has repeated recalls")
        expected_itemids = pres_itemids[trial_index, recalled_positions - 1]
        if not np.array_equal(item_row[:item_count], expected_itemids):
            raise DataContractError(
                f"Trial {trial_index} rec_itemids do not match recalled positions"
            )


def validate_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the structural contract and return a compact summary."""

    arrays = {key: _array(dataset, key) for key in REQUIRED_FIELDS}
    trial_count = arrays["subject"].shape[0]
    if trial_count == 0:
        raise DataContractError("Dataset contains no trials")

    for key in TRIAL_FIELDS:
        value = arrays[key]
        _require_integer(key, value)
        if value.shape != (trial_count, 1):
            raise DataContractError(
                f"{key} must have shape ({trial_count}, 1); found {value.shape}"
            )

    for key in POSITION_FIELDS + RECALL_FIELDS:
        value = arrays[key]
        if value.shape != (trial_count, TARGET_LIST_LENGTH):
            raise DataContractError(
                f"{key} must have shape ({trial_count}, {TARGET_LIST_LENGTH}); "
                f"found {value.shape}"
            )

    for key in (
        "pres_itemnos",
        "pres_itemids",
        "condition",
        "cond3",
        "lpp_imputed",
    ) + RECALL_FIELDS:
        _require_integer(key, arrays[key])

    if not np.all(arrays["listLength"] == TARGET_LIST_LENGTH):
        raise DataContractError("Every listLength must equal 20")
    expected_positions = np.arange(1, TARGET_LIST_LENGTH + 1)
    if not np.all(arrays["pres_itemnos"] == expected_positions):
        raise DataContractError("Every pres_itemnos row must be exactly 1..20")
    if np.any(arrays["pres_itemids"] <= 0):
        raise DataContractError("Every presented item ID must be positive")

    list_types = arrays["list_type"].reshape(-1)
    unexpected_list_types = set(np.unique(list_types)) - set(LIST_TYPE_CODES.values())
    if unexpected_list_types:
        raise DataContractError(
            f"Unexpected list_type values: {sorted(unexpected_list_types)}"
        )

    conditions = arrays["condition"]
    if set(np.unique(conditions)) - set(CONDITION_CODES.values()):
        raise DataContractError("condition may contain only codes 1 and 2")
    if set(np.unique(arrays["cond3"])) - set(COND3_CODES.values()):
        raise DataContractError("cond3 may contain only codes 1, 2, and 3")
    if not np.all(conditions[list_types == LIST_TYPE_PURE_NEGATIVE] == 1):
        raise DataContractError("Pure-negative lists must contain only condition 1")
    if not np.all(conditions[list_types == LIST_TYPE_PURE_NEUTRAL] == 2):
        raise DataContractError("Pure-neutral lists must contain only condition 2")

    pure_negative_cond3 = arrays["cond3"][list_types == LIST_TYPE_PURE_NEGATIVE]
    if pure_negative_cond3.size and not np.all(pure_negative_cond3 == 1):
        raise DataContractError("Pure-negative lists must contain only cond3 code 1")
    for trial_index in np.flatnonzero(list_types == LIST_TYPE_PURE_NEUTRAL):
        values = np.unique(arrays["cond3"][trial_index])
        if values.size != 1 or values[0] not in (2, 3):
            raise DataContractError(
                "Each pure-neutral list must be homogeneous Neutral or Category"
            )

    lpp_imputed = arrays["lpp_imputed"]
    if set(np.unique(lpp_imputed)) - {0, 1}:
        raise DataContractError("lpp_imputed may contain only 0 and 1")
    for key in ("EarlyLPP", "LateLPP"):
        if not np.issubdtype(arrays[key].dtype, np.floating):
            raise DataContractError(f"{key} must have a floating dtype")
        if not np.all(np.isfinite(arrays[key])):
            raise DataContractError(f"{key} contains NaN or infinite values")

    _validate_recall_rows(dataset)

    list_type_counts = Counter(int(value) for value in list_types)
    return {
        "trial_count": trial_count,
        "subject_count": int(np.unique(arrays["subject"]).size),
        "list_type_counts": {
            name: list_type_counts[code] for name, code in LIST_TYPE_CODES.items()
        },
        "list_lengths": sorted(int(value) for value in np.unique(arrays["listLength"])),
        "item_code_max": int(arrays["pres_itemids"].max()),
        "recall_events": int(np.count_nonzero(arrays["recalls"])),
        "lpp_imputed_items": int(lpp_imputed.sum()),
    }


def validate_combined_dataset(
    dataset: Mapping[str, Any],
    *,
    enforce_source_counts: bool = True,
) -> dict[str, Any]:
    """Validate structure plus the accepted mixed/pure source invariants."""

    summary = validate_dataset(dataset)
    list_types = np.asarray(dataset["list_type"]).reshape(-1)
    subjects = np.asarray(dataset["subject"]).reshape(-1)
    lists = np.asarray(dataset["list"]).reshape(-1)

    if list_types.size != EXPECTED_COMBINED_TRIALS:
        raise DataContractError(
            f"Expected {EXPECTED_COMBINED_TRIALS} trials; found {list_types.size}"
        )
    if not np.all(list_types[:EXPECTED_MIXED_TRIALS] == LIST_TYPE_MIXED):
        raise DataContractError("The established 342 mixed rows must remain first")

    expected_counts = {
        "mixed": EXPECTED_MIXED_TRIALS,
        "pure_negative": EXPECTED_PURE_NEGATIVE_TRIALS,
        "pure_neutral": EXPECTED_PURE_NEUTRAL_TRIALS,
    }
    if summary["list_type_counts"] != expected_counts:
        raise DataContractError(
            f"Unexpected list_type counts: {summary['list_type_counts']}"
        )
    if summary["subject_count"] != EXPECTED_SUBJECTS:
        raise DataContractError(
            f"Expected {EXPECTED_SUBJECTS} subjects; found {summary['subject_count']}"
        )

    mixed_subject_count = int(
        np.unique(subjects[list_types == LIST_TYPE_MIXED]).size
    )
    if mixed_subject_count != MIXED_EXPECTED_SUBJECTS:
        raise DataContractError(
            f"Expected {MIXED_EXPECTED_SUBJECTS} mixed-list subjects; "
            f"found {mixed_subject_count}"
        )
    summary["mixed_subject_count"] = mixed_subject_count

    per_subject_counts = Counter(int(subject) for subject in subjects)
    if set(per_subject_counts.values()) != {EXPECTED_LISTS_PER_SUBJECT}:
        raise DataContractError("Every subject must contribute exactly nine lists")
    for subject in np.unique(subjects):
        subject_lists = set(int(value) for value in lists[subjects == subject])
        if subject_lists != set(range(1, EXPECTED_LISTS_PER_SUBJECT + 1)):
            raise DataContractError(
                f"Subject {int(subject)} does not contain list IDs 1..9"
            )

    if enforce_source_counts:
        expected_source_values = {
            "item_code_max": EXPECTED_ITEM_CODE_MAX,
            "lpp_imputed_items": EXPECTED_LPP_IMPUTED_ITEMS,
            "recall_events": EXPECTED_RECALL_EVENTS,
        }
        for key, expected in expected_source_values.items():
            if summary[key] != expected:
                raise DataContractError(
                    f"Expected {key}={expected}; found {summary[key]}"
                )

    return summary
