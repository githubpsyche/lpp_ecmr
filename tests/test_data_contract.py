from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpp_ecmr.data_contract import (
    LEGACY_MIXED_FIELD_DIGESTS,
    LIST_TYPE_CODES,
    MIXED_TRIAL_QUERY,
    DataContractError,
    mixed_trial_mask,
    semantic_array_digest,
    slice_trials,
    validate_combined_dataset,
)
from lpp_ecmr.data_preparation import (
    BuildResult,
    assert_mixed_compatibility,
    prepare_pure_dataframe,
    read_hdf5_dataset,
    write_build,
    write_hdf5_dataset,
)

EXPECTED_LEGACY_MIXED_FIELD_DIGESTS = {
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


def _structural_combined_dataset() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, int]] = []
    for subject in range(202, 240):
        rows.extend((subject, list_id, 0) for list_id in range(1, 10))
    for subject in range(101, 134):
        rows.extend(
            (subject, list_id, 1 if list_id <= 3 else 2) for list_id in range(1, 10)
        )

    trial_count = len(rows)
    subject = np.array([[row[0]] for row in rows], dtype=np.int64)
    lists = np.array([[row[1]] for row in rows], dtype=np.int64)
    list_type = np.array([[row[2]] for row in rows], dtype=np.int64)
    positions = np.tile(np.arange(1, 21, dtype=np.int64), (trial_count, 1))

    condition = np.empty((trial_count, 20), dtype=np.int64)
    cond3 = np.empty_like(condition)
    for index, (_, list_id, code) in enumerate(rows):
        if code == 0:
            condition[index] = np.array([1] * 7 + [2] * 13)
            cond3[index] = np.array([1] * 7 + [2] * 7 + [3] * 6)
        elif code == 1:
            condition[index] = 1
            cond3[index] = 1
        else:
            condition[index] = 2
            cond3[index] = 2 if list_id <= 6 else 3

    recalls = np.zeros((trial_count, 20), dtype=np.int64)
    recalls[:, 0] = 1
    rec_itemids = np.zeros_like(recalls)
    rec_itemids[:, 0] = 1
    return {
        "subject": subject,
        "list": lists,
        "listLength": np.full((trial_count, 1), 20, dtype=np.int64),
        "list_type": list_type,
        "pres_itemnos": positions.copy(),
        "pres_itemids": positions.copy(),
        "recalls": recalls,
        "restricted_recalls": recalls.copy(),
        "rec_itemids": rec_itemids,
        "EarlyLPP": np.zeros((trial_count, 20), dtype=np.float64),
        "LateLPP": np.zeros((trial_count, 20), dtype=np.float64),
        "condition": condition,
        "cond3": cond3,
        "lpp_imputed": np.zeros((trial_count, 20), dtype=np.int64),
    }


def test_approved_list_type_contract_is_frozen():
    assert LIST_TYPE_CODES == {
        "mixed": 0,
        "pure_negative": 1,
        "pure_neutral": 2,
    }
    assert MIXED_TRIAL_QUERY == "data['list_type'] == 0"


def test_legacy_mixed_semantic_digests_are_explicit_and_frozen():
    assert LEGACY_MIXED_FIELD_DIGESTS == EXPECTED_LEGACY_MIXED_FIELD_DIGESTS
    assert semantic_array_digest(np.array([[1, 2]], dtype=np.int64)) == (
        "c38758e0eb2eeeab31588c9a91e96d14c438d56407d1417e014304b7d63cfa0c"
    )


def test_combined_contract_accepts_342_99_198_structure():
    dataset = _structural_combined_dataset()
    summary = validate_combined_dataset(dataset, enforce_source_counts=False)
    assert summary["trial_count"] == 639
    assert summary["subject_count"] == 71
    assert summary["mixed_subject_count"] == 38
    assert summary["list_type_counts"] == {
        "mixed": 342,
        "pure_negative": 99,
        "pure_neutral": 198,
    }
    mask = mixed_trial_mask(dataset)
    assert mask.shape == (639,)
    assert mask.sum() == 342
    mixed = slice_trials(dataset, mask)
    assert mixed["subject"].shape == (342, 1)
    assert np.all(mixed["list_type"] == 0)


def test_combined_contract_rejects_wrong_mixed_subject_count():
    dataset = _structural_combined_dataset()
    first_mixed = 0
    first_pure = 342
    dataset["subject"][[first_mixed, first_pure]] = dataset["subject"][
        [first_pure, first_mixed]
    ]

    with pytest.raises(
        DataContractError,
        match="Expected 38 mixed-list subjects; found 39",
    ):
        validate_combined_dataset(dataset, enforce_source_counts=False)


def test_mixed_trial_mask_requires_list_type():
    dataset = _structural_combined_dataset()
    legacy = {key: value[:342] for key, value in dataset.items() if key != "list_type"}
    with pytest.raises(DataContractError, match="does not contain list_type"):
        mixed_trial_mask(legacy)
    legacy["bad"] = np.zeros((341, 1))
    with pytest.raises(DataContractError, match="not aligned"):
        slice_trials(legacy, np.ones(342, dtype=bool))


def test_combined_contract_rejects_pure_condition_mismatch():
    dataset = _structural_combined_dataset()
    first_pure_negative = 342
    dataset["condition"][first_pure_negative, 0] = 2
    with pytest.raises(DataContractError, match="Pure-negative"):
        validate_combined_dataset(dataset, enforce_source_counts=False)


def test_combined_contract_rejects_recalled_item_mismatch():
    dataset = _structural_combined_dataset()
    dataset["rec_itemids"][0, 0] = 2
    with pytest.raises(DataContractError, match="rec_itemids"):
        validate_combined_dataset(dataset, enforce_source_counts=False)


def test_hdf5_round_trip_and_mixed_compatibility(tmp_path: Path):
    dataset = _structural_combined_dataset()
    path = tmp_path / "combined.h5"
    write_hdf5_dataset(dataset, path)
    loaded = read_hdf5_dataset(path)
    for key in dataset:
        assert np.array_equal(loaded[key], dataset[key])

    baseline = {
        key: value[:342] for key, value in dataset.items() if key != "list_type"
    }
    assert_mixed_compatibility(dataset, baseline)
    baseline["recalls"] = baseline["recalls"].copy()
    baseline["recalls"][0, 0] = 2
    with pytest.raises(DataContractError, match="recalls"):
        assert_mixed_compatibility(dataset, baseline)


def test_write_build_requires_an_explicit_metadata_sidecar(tmp_path: Path):
    dataset = _structural_combined_dataset()
    result = BuildResult(dataset=dataset, metadata={"summary": {"trial_count": 639}})
    output = tmp_path / "combined.h5"

    written, metadata = write_build(result, output)

    assert written == output
    assert metadata is None
    assert output.exists()
    assert not output.with_suffix(".metadata.json").exists()

    optional_metadata = tmp_path / "rebuild-details.json"
    write_build(
        result,
        output,
        metadata_output=optional_metadata,
        force=True,
    )
    assert optional_metadata.exists()


def test_pure_alignment_derives_positions_instead_of_trusting_source(
    tmp_path: Path,
):
    behavior_dir = tmp_path / "behavior"
    behavior_dir.mkdir()
    behavior_rows: list[list[object]] = [
        [1, 1, "Buffer", 99, "buffer1.jpg"],
        [1, 2, "Buffer", 99, "buffer2.jpg"],
    ]
    trial_numbers = list(range(3, 23))
    trial_numbers[11] = 199
    for position, trial_number in enumerate(trial_numbers, start=1):
        behavior_rows.append(
            [
                1,
                trial_number,
                "Negative",
                1 if position % 2 else 0,
                f"neg{position}.jpg",
            ]
        )
    pd.DataFrame(behavior_rows).to_csv(
        behavior_dir / "HT_101.csv",
        header=False,
        index=False,
    )

    eeg_rows: list[list[object]] = []
    for position, trial_number in enumerate(trial_numbers, start=1):
        if position == 5:
            continue
        source_position = 99 if trial_number == 199 else position
        eeg_rows.append(
            [
                101,
                1,
                "Negative",
                "Negative",
                trial_number,
                source_position,
                f"neg{position}",
                1 if position % 2 else 0,
                float(position),
                float(position + 1),
            ]
        )
    eeg_path = tmp_path / "pure_eeg.csv"
    pd.DataFrame(eeg_rows).to_csv(eeg_path, header=False, index=False)

    frame, diagnostics, _ = prepare_pure_dataframe(
        behavior_dir,
        eeg_path,
        subject_ids=(101,),
        expected_lists=1,
    )
    assert frame["PresentationOrder"].tolist() == list(range(1, 21))
    anomaly = frame.loc[frame["TrialNumber"] == 199].iloc[0]
    assert anomaly["PresentationOrder"] == 12
    assert anomaly["EEGPresentationOrder"] == 99
    assert diagnostics["eeg_rows_source_position_99"] == 1
    assert diagnostics["lpp_missing_events"] == 1
    assert frame["lpp_imputed"].sum() == 1
