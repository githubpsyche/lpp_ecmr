"""Tests for authenticated Early-LPP empirical source fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from work.pooled_model_figures import (
    build_original_early_lpp_summaries as build,
)


def _write_extracted_source(root: Path) -> bytes:
    empirical = b"empirical rows\n"
    empirical_path = root / build.EMPIRICAL_DATA_MEMBER
    empirical_path.parent.mkdir(parents=True)
    empirical_path.write_bytes(empirical)
    return empirical


def test_extracted_delivery_accepts_expected_csv_hash(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    expected_empirical = _write_extracted_source(extracted)

    empirical = build._load_empirical_source_bytes(
        extracted_root=extracted,
        expected_sha256=hashlib.sha256(expected_empirical).hexdigest(),
    )

    assert empirical == expected_empirical


def test_extracted_delivery_rejects_changed_csv(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    original = _write_extracted_source(extracted)
    (extracted / build.EMPIRICAL_DATA_MEMBER).write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build._load_empirical_source_bytes(
            extracted_root=extracted,
            expected_sha256=hashlib.sha256(original).hexdigest(),
        )
