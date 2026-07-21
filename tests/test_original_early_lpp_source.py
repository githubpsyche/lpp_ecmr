"""Tests for authenticated Early-LPP empirical source fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from work.pooled_model_figures import (
    build_original_early_lpp_summaries as build,
)


def _write_extracted_source(
    root: Path,
    provenance_path: Path,
) -> tuple[bytes, bytes]:
    empirical = b"empirical rows\n"
    analysis = b"analysis source\n"
    empirical_path = root / build.EMPIRICAL_DATA_MEMBER
    analysis_path = root / build.EMPIRICAL_CODE_MEMBER
    empirical_path.parent.mkdir(parents=True)
    analysis_path.parent.mkdir(parents=True)
    empirical_path.write_bytes(empirical)
    analysis_path.write_bytes(analysis)
    provenance_path.write_text(
        json.dumps(
            {
                "archive": "/delivery/original.zip",
                "archive_sha256": "a" * 64,
                "data_member": build.EMPIRICAL_DATA_MEMBER,
                "data_member_sha256": hashlib.sha256(empirical).hexdigest(),
                "analysis_member": build.EMPIRICAL_CODE_MEMBER,
                "analysis_member_sha256": hashlib.sha256(analysis).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return empirical, analysis


def test_extracted_delivery_fallback_requires_recorded_member_hashes(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    provenance = tmp_path / "source.json"
    expected_empirical, expected_analysis = _write_extracted_source(
        extracted,
        provenance,
    )

    empirical, analysis, metadata = build._load_empirical_source_bytes(
        archive_path=tmp_path / "absent.zip",
        extracted_root=extracted,
        provenance_path=provenance,
    )

    assert empirical == expected_empirical
    assert analysis == expected_analysis
    assert metadata["source_type"] == "extracted_directory"
    assert metadata["source_path"] == str(extracted)
    assert metadata["archive"] == "/delivery/original.zip"


def test_extracted_delivery_fallback_rejects_changed_member(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    provenance = tmp_path / "source.json"
    _write_extracted_source(extracted, provenance)
    (extracted / build.EMPIRICAL_DATA_MEMBER).write_bytes(b"changed")

    with pytest.raises(ValueError, match="data_member_sha256 mismatch"):
        build._load_empirical_source_bytes(
            archive_path=tmp_path / "absent.zip",
            extracted_root=extracted,
            provenance_path=provenance,
        )
