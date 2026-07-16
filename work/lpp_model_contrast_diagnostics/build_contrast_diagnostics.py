#!/usr/bin/env python3
"""Build a contrast-first diagnostic figure without changing the cell-means package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
SOURCE_PACKAGE_DIR = PROJECT_ROOT / "work" / "lpp_model_results"
SOURCE_BUILDER_PATH = SOURCE_PACKAGE_DIR / "build_results.py"
FIGURE_SCRIPT_PATH = PACKAGE_DIR / "build_contrast_figure.R"

FIGURE_MODEL_IDS = (
    "CategoryOnly_eCMR_LPP_EmotionalOnly",
    "EEM_eCMR",
    "EEM_eCMR_LPP_General",
    "EEM_eCMR_LPP_EmotionalOnly",
)
SOURCE_ORDER = ("Observed",) + FIGURE_MODEL_IDS


def _load_source_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lpp_model_results_source_builder", SOURCE_BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = _load_source_builder()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "source_type",
        "source_id",
        "source_label",
        "diagnostic_id",
        "diagnostic_label",
        "estimate",
        "lower",
        "upper",
        "interval",
        "confidence_level",
        "n_units",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _contrast_values(
    recall_rates: np.ndarray, lpp_means: np.ndarray
) -> dict[str, np.ndarray]:
    if recall_rates.ndim != 2 or recall_rates.shape[1] != 2:
        raise ValueError(f"Unexpected recall summary shape: {recall_rates.shape}")
    if lpp_means.ndim != 3 or lpp_means.shape[1:] != (2, 2):
        raise ValueError(f"Unexpected LPP summary shape: {lpp_means.shape}")
    values = {
        "recall_gap": recall_rates[:, 0] - recall_rates[:, 1],
        "lpp_negative": lpp_means[:, 0, 0] - lpp_means[:, 0, 1],
        "lpp_neutral": lpp_means[:, 1, 0] - lpp_means[:, 1, 1],
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Contrast summaries contain missing or non-finite values")
    return values


DIAGNOSTIC_LABELS = {
    "recall_gap": "Negative - Neutral recall rate",
    "lpp_negative": "Recalled - Unrecalled Early LPP: Negative",
    "lpp_neutral": "Recalled - Unrecalled Early LPP: Neutral",
}


def _summary_rows(
    *,
    source_type: str,
    source_id: str,
    source_label: str,
    values: dict[str, np.ndarray],
    bootstrap: bool,
) -> list[dict[str, Any]]:
    alpha = (1 - source.CONFIDENCE_LEVEL) / 2
    if bootstrap:
        unit_count = len(next(iter(values.values())))
        rng = np.random.default_rng(source.BOOTSTRAP_SEED)
        indices = rng.integers(
            0,
            unit_count,
            size=(source.BOOTSTRAP_SAMPLES, unit_count),
        )
        interval_label = "Participant percentile bootstrap"
    else:
        indices = None
        interval_label = "Central interval across complete simulated datasets"

    rows: list[dict[str, Any]] = []
    for diagnostic_id, unit_values in values.items():
        interval_values = (
            np.mean(unit_values[indices], axis=1)
            if indices is not None
            else unit_values
        )
        lower, upper = np.quantile(interval_values, [alpha, 1 - alpha])
        rows.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "source_label": source_label,
                "diagnostic_id": diagnostic_id,
                "diagnostic_label": DIAGNOSTIC_LABELS[diagnostic_id],
                "estimate": float(np.mean(unit_values)),
                "lower": float(lower),
                "upper": float(upper),
                "interval": interval_label,
                "confidence_level": source.CONFIDENCE_LEVEL,
                "n_units": len(unit_values),
            }
        )
    return rows


def build_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    run_manifest = source._load_json(source.RUN_MANIFEST_PATH)
    expected_lists = int(run_manifest["fit_settings"]["expected_list_count"])
    expected_simulations = int(run_manifest["fit_settings"]["experiment_count"])

    observed_data = source._load_recall_data(source.DATA_PATH)
    subject_ids, observed_recall, observed_lpp = source.summarize_observed(
        observed_data
    )
    if len(subject_ids) != 38:
        raise ValueError(f"Expected 38 participants, found {len(subject_ids)}")

    rows = _summary_rows(
        source_type="observed",
        source_id="Observed",
        source_label="Observed data",
        values=_contrast_values(observed_recall, observed_lpp),
        bootstrap=True,
    )
    simulation_paths: list[Path] = []
    for model_id in FIGURE_MODEL_IDS:
        model = source.MODEL_BY_ID[model_id]
        _, simulation_path = source._fit_paths(model)
        simulation_paths.append(simulation_path)
        simulated_data = source._load_recall_data(simulation_path)
        simulated_recall, simulated_lpp = source.summarize_simulations(
            simulated_data,
            expected_lists,
            expected_simulations,
        )
        rows.extend(
            _summary_rows(
                source_type="predicted",
                source_id=model_id,
                source_label=model.manuscript_label,
                values=_contrast_values(simulated_recall, simulated_lpp),
                bootstrap=False,
            )
        )

    if len(rows) != len(SOURCE_ORDER) * len(DIAGNOSTIC_LABELS):
        raise AssertionError(f"Unexpected contrast-row count: {len(rows)}")
    return rows, simulation_paths


def _write_caption() -> None:
    alt = (
        "Three horizontal-bar panels compare observed data with four model "
        "predictions. The first panel shows the Negative-minus-Neutral recall-rate "
        "contrast. The next panels show Recalled-minus-Unrecalled Early-LPP "
        "contrasts for Negative and Neutral items. Models with categorical "
        "enhancement reproduce the recall-rate contrast, whereas models with LPP "
        "modulation reproduce the Negative-item Early-LPP contrast."
    )
    caption = (
        "Contrast-first diagnostics for the observed data and four selected pooled "
        "source-context model predictions. Positive recall-rate values indicate "
        "higher recall for Negative than Neutral items. Positive Early-LPP values "
        "indicate higher mean Early LPP for Recalled than Unrecalled items. Observed "
        "error bars are percentile 95% confidence intervals calculated from 10,000 "
        "participant-level bootstrap samples ($N=38$). Predicted error bars are "
        "central 95% intervals across 200 complete simulated datasets and represent "
        "simulation variability rather than parameter-estimation uncertainty. The "
        "simulations condition on each list's observed recall count."
    )
    fragment = (
        f'![{caption}](contrast_diagnostic_figure.svg)'
        f'{{#fig-model-contrast-diagnostics width="100%" fig-alt="{alt}"}}\n'
    )
    (PACKAGE_DIR / "contrast_diagnostic_figure_caption.qmd").write_text(
        fragment, encoding="utf-8"
    )
    (PACKAGE_DIR / "contrast_diagnostic_figure_alt.txt").write_text(
        alt + "\n", encoding="utf-8"
    )


def _command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def _write_manifest(
    source_paths: list[Path], generated_paths: list[Path]
) -> None:
    manifest = {
        "package": "work/lpp_model_contrast_diagnostics",
        "purpose": "Contrast-first alternative to the retained cell-means diagnostic figure",
        "source_order": list(SOURCE_ORDER),
        "diagnostics": DIAGNOSTIC_LABELS,
        "source_artifacts": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
        "generated_artifacts": {
            path.name: _sha256(path) for path in generated_paths
        },
        "summary_settings": {
            "observed_unit": "participant contrast",
            "observed_units": 38,
            "observed_interval": "percentile bootstrap of participant contrasts",
            "bootstrap_samples": source.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": source.BOOTSTRAP_SEED,
            "simulation_unit": "complete simulated-dataset contrast",
            "simulation_units_per_model": 200,
            "simulation_interval": "central interval across simulated-dataset contrasts",
            "confidence_level": source.CONFIDENCE_LEVEL,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "R": _command_version(["Rscript", "--version"]),
            "ggplot2": _command_version(
                [
                    "Rscript",
                    "-e",
                    'cat(as.character(packageVersion("ggplot2")))',
                ]
            ),
            "pdftocairo": _command_version(["pdftocairo", "-v"]),
        },
    }
    (PACKAGE_DIR / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows, simulation_paths = build_rows()
    summary_path = PACKAGE_DIR / "contrast_summary.csv"
    _write_csv(summary_path, rows)
    _write_caption()
    subprocess.run(
        ["Rscript", str(FIGURE_SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    generated_paths = [
        summary_path,
        PACKAGE_DIR / "contrast_diagnostic_figure.svg",
        PACKAGE_DIR / "contrast_diagnostic_figure.pdf",
        PACKAGE_DIR / "contrast_diagnostic_figure.png",
        PACKAGE_DIR / "contrast_diagnostic_figure_caption.qmd",
        PACKAGE_DIR / "contrast_diagnostic_figure_alt.txt",
    ]
    source_paths = [
        source.DATA_PATH,
        source.RUN_MANIFEST_PATH,
        SOURCE_BUILDER_PATH,
        Path(__file__),
        FIGURE_SCRIPT_PATH,
        *simulation_paths,
    ]
    _write_manifest(source_paths, generated_paths)
    print(
        json.dumps(
            {
                "package": PACKAGE_DIR.relative_to(PROJECT_ROOT).as_posix(),
                "sources": len(SOURCE_ORDER),
                "contrasts": len(rows),
                "figure": "contrast_diagnostic_figure.pdf",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
