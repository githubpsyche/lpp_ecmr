#!/usr/bin/env python3
"""Build the six-model result tables and grouped-bar diagnostics for Issue #10.

The script reads the returned pooled fit JSON and simulation HDF5 artifacts from
``work/lpp_model_comparison``.  It does not copy or modify those source files.
All products are written beside this script so that the result package remains
flat, bounded, and reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
FIT_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"
RUN_MANIFEST_PATH = FIT_DIR / "run_manifest.json"
FIGURE_SCRIPT_PATH = PACKAGE_DIR / "build_diagnostic_figure.R"
ORIGINAL_LPP_CONTRAST_PATH = (
    PROJECT_ROOT
    / "work"
    / "lpp_model_prediction_grids"
    / "original_early_lpp_contrasts.csv"
)

BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

NEGATIVE = 1
NEUTRAL = 2
CATEGORY_ORDER = (NEGATIVE, NEUTRAL)
CATEGORY_LABELS = {NEGATIVE: "Negative", NEUTRAL: "Neutral"}
RECALL_ORDER = (True, False)
RECALL_LABELS = {True: "Remembered", False: "Forgotten"}

@dataclass(frozen=True)
class ModelSpec:
    registry_id: str
    categorical_enhancement: str
    lpp_modulation: str
    short_code: str

    @property
    def manuscript_label(self) -> str:
        enhancement = (
            "Categorical enhancement"
            if self.categorical_enhancement == "Present"
            else "No categorical enhancement"
        )
        return f"{enhancement} + {self.lpp_modulation} LPP"

    @property
    def parameter_table_label(self) -> str:
        emotion_boost = (
            "Emotion-based learning boost"
            if self.categorical_enhancement == "Present"
            else "No emotion-based learning boost"
        )
        lpp_boost = {
            "No": "no LPP-based learning boost",
            "General": "LPP-based learning boost for all items",
            "Emotion-dependent": (
                "LPP-based learning boost for negative items only"
            ),
        }[self.lpp_modulation]
        return f"{emotion_boost} + {lpp_boost}"

    @property
    def row_label(self) -> str:
        enhancement = f"Enhancement {self.categorical_enhancement.lower()}"
        return f"{enhancement}\n{self.lpp_modulation} LPP"


MODELS = (
    ModelSpec("CategoryOnly_eCMR", "Absent", "No", "A"),
    ModelSpec("CategoryOnly_eCMR_LPP_General", "Absent", "General", "B"),
    ModelSpec(
        "CategoryOnly_eCMR_LPP_EmotionalOnly",
        "Absent",
        "Emotion-dependent",
        "C",
    ),
    ModelSpec("EEM_eCMR", "Present", "No", "D"),
    ModelSpec("EEM_eCMR_LPP_General", "Present", "General", "E"),
    ModelSpec(
        "EEM_eCMR_LPP_EmotionalOnly",
        "Present",
        "Emotion-dependent",
        "F",
    ),
)

MODEL_BY_ID = {model.registry_id: model for model in MODELS}
EMOTION_BOOST_LABELS = {
    "Absent": "No",
    "Present": "Yes",
}
LPP_BOOST_LABELS = {
    "No": "None",
    "General": "All items",
    "Emotion-dependent": "Negative items only",
}
MODEL_SPECIFICATION_LABELS = {
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
FIGURE_MODEL_IDS = (
    "CategoryOnly_eCMR_LPP_EmotionalOnly",
    "EEM_eCMR",
    "EEM_eCMR_LPP_General",
    "EEM_eCMR_LPP_EmotionalOnly",
)

PARAMETERS = (
    (
        "encoding_drift_rate",
        r"$\beta_{\mathrm{enc}}^{T}$",
        "Temporal-context drift at study",
    ),
    (
        "source_encoding_drift_rate",
        r"$\beta_{\mathrm{enc}}^{S}$",
        "Source-context drift at study",
    ),
    (
        "start_drift_rate",
        r"$\beta_{\mathrm{start}}$",
        "Pre-recall drift toward start-of-list context",
    ),
    (
        "recall_drift_rate",
        r"$\beta_{\mathrm{rec}}^{T}$",
        "Temporal-context reinstatement after recall",
    ),
    (
        "source_recall_drift_rate",
        r"$\beta_{\mathrm{rec}}^{S}$",
        "Source-context reinstatement after recall",
    ),
    (
        "shared_support",
        r"$\alpha$",
        "Shared pre-experimental context-to-item support",
    ),
    (
        "item_support",
        r"$\delta$",
        "Item-specific pre-experimental context-to-item support",
    ),
    ("learning_rate", r"$\gamma$", "Feature-to-context learning rate"),
    ("primacy_scale", r"$\phi_s$", "Primacy scale"),
    ("primacy_decay", r"$\phi_d$", "Primacy decay"),
    ("choice_sensitivity", r"$\tau_c$", "Recall-choice sensitivity"),
    (
        "source_learning_rate",
        r"$\omega$",
        "Source learning relative to temporal learning",
    ),
    (
        "emotion_scale",
        r"$\phi_{\mathrm{emot}}$",
        "Categorical multiplier for negative items",
    ),
    (
        "lpp_main_scale",
        r"$\kappa$",
        "General Early-LPP slope on log learning strength",
    ),
    (
        "lpp_inter_scale",
        r"$\kappa_{\mathrm{emot}}$",
        "Emotion-dependent Early-LPP slope increment",
    ),
)

PARAMETER_RANGES = {
    "encoding_drift_rate": "$(0,1)$",
    "source_encoding_drift_rate": "$(0,1)$",
    "start_drift_rate": "$(0,1)$",
    "recall_drift_rate": "$(0,1)$",
    "source_recall_drift_rate": "$1$ (fixed)",
    "shared_support": "$(0,100]$",
    "item_support": "$(0,100]$",
    "learning_rate": "$(0,1)$",
    "primacy_scale": "$(0,100]$",
    "primacy_decay": "$(0,100]$",
    "choice_sensitivity": "$(0,100]$",
    "source_learning_rate": "$(0,10]$",
    "emotion_scale": "$1$ (fixed) or $(0,10]$",
    "lpp_main_scale": "$0$ (fixed) or $[0,0.2145]$",
    "lpp_inter_scale": "$0$ (fixed) or $[0,0.2145]$",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(value: Any) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one fitted value, got {value!r}")
        value = value[0]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite fitted value: {result}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _fit_paths(model: ModelSpec) -> tuple[Path, Path]:
    stem = f"{model.registry_id}_best_of_3"
    return FIT_DIR / f"{stem}.json", FIT_DIR / f"{stem}.h5"


def load_fit_results() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        fit_path, simulation_path = _fit_paths(model)
        if not fit_path.exists() or not simulation_path.exists():
            raise FileNotFoundError(
                f"Missing returned artifact for {model.registry_id}: "
                f"{fit_path.name} or {simulation_path.name}"
            )
        result = _load_json(fit_path)
        nll = _scalar(result["fitness"])
        free_parameters = result["free"]
        parameter_count = len(free_parameters)
        aic = 2 * nll + 2 * parameter_count
        converged = result.get("converged", [False])
        if not bool(converged[0] if isinstance(converged, list) else converged):
            raise ValueError(f"Selected fit did not converge: {model.registry_id}")
        for parameter, bounds in free_parameters.items():
            estimate = _scalar(result["fits"][parameter])
            lower, upper = map(float, bounds)
            if not lower <= estimate <= upper:
                raise ValueError(
                    f"{model.registry_id} {parameter}={estimate} outside {bounds}"
                )
        rows.append(
            {
                "model_code": model.short_code,
                "model": model.manuscript_label,
                "registry_id": model.registry_id,
                "categorical_enhancement": model.categorical_enhancement,
                "lpp_modulation": model.lpp_modulation,
                "free_parameters": parameter_count,
                "NLL": nll,
                "AIC": aic,
            }
        )
        results[model.registry_id] = result

    best_aic = min(float(row["AIC"]) for row in rows)
    for row in rows:
        row["delta_AIC"] = float(row["AIC"]) - best_aic
        row["Akaike_weight"] = math.exp(-0.5 * float(row["delta_AIC"]))
    weight_sum = sum(float(row["Akaike_weight"]) for row in rows)
    for row in rows:
        row["Akaike_weight"] = float(row["Akaike_weight"]) / weight_sum
    return rows, results


def parameter_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        result = results[model.registry_id]
        free = result["free"]
        fixed = result["fixed"]
        fits = result["fits"]
        for implementation_name, symbol, description in PARAMETERS:
            if implementation_name not in fits:
                raise KeyError(
                    f"{model.registry_id} lacks expected parameter {implementation_name}"
                )
            estimate = _scalar(fits[implementation_name])
            status = "free" if implementation_name in free else "fixed"
            if status == "fixed" and implementation_name not in fixed:
                raise KeyError(
                    f"{model.registry_id} does not declare fixed {implementation_name}"
                )
            bounds = free.get(implementation_name)
            rows.append(
                {
                    "model_code": model.short_code,
                    "model": model.parameter_table_label,
                    "registry_id": model.registry_id,
                    "implementation_parameter": implementation_name,
                    "manuscript_symbol": symbol.replace("$", ""),
                    "description": description,
                    "estimate": estimate,
                    "status": status,
                    "lower_bound": "" if bounds is None else float(bounds[0]),
                    "upper_bound": "" if bounds is None else float(bounds[1]),
                }
            )
    return rows


def _recall_hits(recalls: np.ndarray, presentations: np.ndarray) -> np.ndarray:
    """Mark each valid presentation that occurs at least once in recalls."""
    hits = np.zeros_like(presentations, dtype=bool)
    for event in range(recalls.shape[0]):
        recalled_position = recalls[event]
        hits |= (recalled_position[None, :] > 0) & (
            presentations == recalled_position[None, :]
        )
    return hits


def _load_recall_data(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["data"]
        data = {
            key: np.asarray(group[key][:])
            for key in (
                "subject",
                "list",
                "listLength",
                "condition",
                "pres_itemnos",
                "recalls",
                "EarlyLPP",
            )
        }
    columns = data["condition"].shape[1]
    if data["condition"].shape[0] != 20:
        raise ValueError(f"Expected 20 study positions in {path}")
    for key in ("pres_itemnos", "recalls", "EarlyLPP"):
        if data[key].shape != (20, columns):
            raise ValueError(f"Unexpected {key} shape in {path}: {data[key].shape}")
    if data["subject"].shape != (1, columns):
        raise ValueError(f"Unexpected subject shape in {path}: {data['subject'].shape}")
    values = set(np.unique(data["condition"]).tolist())
    if not values.issubset({0, NEGATIVE, NEUTRAL}):
        raise ValueError(f"Unexpected condition values in {path}: {values}")
    return data


def _center_lpp(lpp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, lpp.astype(float), np.nan)
    centered = masked - np.nanmean(masked, axis=0, keepdims=True)
    max_abs_column_mean = float(np.nanmax(np.abs(np.nanmean(centered, axis=0))))
    if max_abs_column_mean > 1e-10:
        raise AssertionError(f"Within-list LPP centering failed: {max_abs_column_mean}")
    return centered


def summarize_observed(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    presentations = data["pres_itemnos"]
    conditions = data["condition"]
    valid = presentations > 0
    hits = _recall_hits(data["recalls"], presentations)
    centered_lpp = _center_lpp(data["EarlyLPP"], valid)
    subjects = data["subject"][0]
    subject_ids = np.unique(subjects)
    recall_numerators = np.zeros((len(subject_ids), 2), dtype=float)
    recall_denominators = np.zeros((len(subject_ids), 2), dtype=float)
    lpp_sums = np.zeros((len(subject_ids), 2, 2), dtype=float)
    lpp_counts = np.zeros((len(subject_ids), 2, 2), dtype=float)

    for subject_index, subject_id in enumerate(subject_ids):
        columns = subjects == subject_id
        for category_index, category in enumerate(CATEGORY_ORDER):
            category_mask = valid[:, columns] & (
                conditions[:, columns] == category
            )
            category_hits = hits[:, columns][category_mask]
            recall_numerators[subject_index, category_index] = float(
                np.sum(category_hits)
            )
            recall_denominators[subject_index, category_index] = float(
                category_hits.size
            )
            for recall_index, remembered in enumerate(RECALL_ORDER):
                cell_mask = category_mask & (
                    hits[:, columns] == remembered
                )
                cell_values = centered_lpp[:, columns][cell_mask]
                lpp_sums[subject_index, category_index, recall_index] = float(
                    np.sum(cell_values)
                )
                lpp_counts[subject_index, category_index, recall_index] = float(
                    cell_values.size
                )
    if (
        not np.isfinite(recall_numerators).all()
        or not np.isfinite(recall_denominators).all()
        or not np.isfinite(lpp_sums).all()
        or not np.isfinite(lpp_counts).all()
        or np.any(recall_denominators == 0)
        or np.any(lpp_counts == 0)
    ):
        raise ValueError("Observed sufficient statistics contain invalid cells")
    return (
        subject_ids,
        recall_numerators,
        recall_denominators,
        lpp_sums,
        lpp_counts,
    )


def summarize_simulations(
    data: dict[str, np.ndarray],
    expected_lists: int,
    expected_simulations: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns = data["condition"].shape[1]
    if columns != expected_lists * expected_simulations:
        raise ValueError(
            f"Expected {expected_lists * expected_simulations} simulated columns, "
            f"found {columns}"
        )
    presentations = data["pres_itemnos"]
    conditions = data["condition"]
    valid = presentations > 0
    hits = _recall_hits(data["recalls"], presentations)
    centered_lpp = _center_lpp(data["EarlyLPP"], valid)

    def reshape(values: np.ndarray) -> np.ndarray:
        return values.reshape(values.shape[0], expected_lists, expected_simulations)

    presentations_3d = reshape(presentations)
    conditions_3d = reshape(conditions)
    valid_3d = reshape(valid)
    hits_3d = reshape(hits)
    centered_lpp_3d = reshape(centered_lpp)

    # The simulator stores 200 realizations consecutively for each original
    # list. Study inputs must therefore be invariant along the final axis.
    for name, values in (
        ("presentations", presentations_3d),
        ("conditions", conditions_3d),
        ("EarlyLPP", centered_lpp_3d),
    ):
        reference = values[:, :, :1]
        if np.issubdtype(values.dtype, np.floating):
            matches = np.allclose(values, reference, equal_nan=True)
        else:
            matches = bool(np.all(values == reference))
        if not matches:
            raise ValueError(f"Simulation storage order check failed for {name}")

    recall_rates = np.full((expected_simulations, 2), np.nan)
    lpp_means = np.full((expected_simulations, 2, 2), np.nan)
    for simulation in range(expected_simulations):
        for category_index, category in enumerate(CATEGORY_ORDER):
            category_mask = valid_3d[:, :, simulation] & (
                conditions_3d[:, :, simulation] == category
            )
            recall_rates[simulation, category_index] = float(
                np.mean(hits_3d[:, :, simulation][category_mask])
            )
            for recall_index, remembered in enumerate(RECALL_ORDER):
                cell_mask = category_mask & (
                    hits_3d[:, :, simulation] == remembered
                )
                lpp_means[simulation, category_index, recall_index] = float(
                    np.mean(centered_lpp_3d[:, :, simulation][cell_mask])
                )
    if not np.isfinite(recall_rates).all() or not np.isfinite(lpp_means).all():
        raise ValueError("Simulated summaries contain missing or non-finite cells")
    return recall_rates, lpp_means


def _observed_intervals(
    recall_numerators: np.ndarray,
    recall_denominators: np.ndarray,
    lpp_sums: np.ndarray,
    lpp_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        recall_numerators.shape[0],
        size=(BOOTSTRAP_SAMPLES, recall_numerators.shape[0]),
    )
    bootstrap_recall = (
        np.sum(recall_numerators[indices], axis=1)
        / np.sum(recall_denominators[indices], axis=1)
    )
    bootstrap_lpp = (
        np.sum(lpp_sums[indices], axis=1)
        / np.sum(lpp_counts[indices], axis=1)
    )
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    recall_bounds = np.quantile(bootstrap_recall, [alpha, 1 - alpha], axis=0)
    lpp_bounds = np.quantile(bootstrap_lpp, [alpha, 1 - alpha], axis=0)
    return (
        np.sum(recall_numerators, axis=0)
        / np.sum(recall_denominators, axis=0),
        recall_bounds,
        np.sum(lpp_sums, axis=0) / np.sum(lpp_counts, axis=0),
        lpp_bounds,
    )


def _simulation_intervals(
    recall_rates: np.ndarray, lpp_means: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return (
        np.mean(recall_rates, axis=0),
        np.quantile(recall_rates, [alpha, 1 - alpha], axis=0),
        np.mean(lpp_means, axis=0),
        np.quantile(lpp_means, [alpha, 1 - alpha], axis=0),
    )


def diagnostic_rows(
    run_manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
]:
    expected_lists = int(run_manifest["fit_settings"]["expected_list_count"])
    expected_simulations = int(run_manifest["fit_settings"]["experiment_count"])
    observed = _load_recall_data(DATA_PATH)
    (
        subject_ids,
        observed_recall_numerators,
        observed_recall_denominators,
        observed_lpp_sums,
        observed_lpp_counts,
    ) = summarize_observed(observed)
    if len(subject_ids) != int(run_manifest["data"]["subject_count"]):
        raise ValueError("Observed subject count does not match run manifest")
    if observed_recall_numerators.shape[0] != 38:
        raise ValueError("Expected 38 participant summaries")

    summaries: dict[str, dict[str, np.ndarray]] = {}
    obs_recall_mean, obs_recall_bounds, obs_lpp_mean, obs_lpp_bounds = (
        _observed_intervals(
            observed_recall_numerators,
            observed_recall_denominators,
            observed_lpp_sums,
            observed_lpp_counts,
        )
    )
    summaries["Observed"] = {
        "recall_mean": obs_recall_mean,
        "recall_bounds": obs_recall_bounds,
        "lpp_mean": obs_lpp_mean,
        "lpp_bounds": obs_lpp_bounds,
    }

    for model in MODELS:
        _, simulation_path = _fit_paths(model)
        simulated = _load_recall_data(simulation_path)
        recall_rates, lpp_means = summarize_simulations(
            simulated, expected_lists, expected_simulations
        )
        recall_mean, recall_bounds, lpp_mean, lpp_bounds = _simulation_intervals(
            recall_rates, lpp_means
        )
        summaries[model.registry_id] = {
            "recall_mean": recall_mean,
            "recall_bounds": recall_bounds,
            "lpp_mean": lpp_mean,
            "lpp_bounds": lpp_bounds,
        }

    rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    sources: list[tuple[str, str, str, str, int]] = [
        (
            "observed",
            "Observed",
            "Observed data",
            "Participant-cluster percentile bootstrap",
            len(subject_ids),
        )
    ] + [
        (
            "predicted",
            model.registry_id,
            model.manuscript_label,
            "Central interval across complete simulated datasets",
            expected_simulations,
        )
        for model in MODELS
    ]
    for source_type, source_id, label, interval, n_units in sources:
        summary = summaries[source_id]
        for category_index, category in enumerate(CATEGORY_ORDER):
            rows.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_label": label,
                    "metric": "recall_rate",
                    "category": CATEGORY_LABELS[category],
                    "recall_status": "",
                    "mean": float(summary["recall_mean"][category_index]),
                    "lower": float(summary["recall_bounds"][0, category_index]),
                    "upper": float(summary["recall_bounds"][1, category_index]),
                    "interval": interval,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "n_units": n_units,
                }
            )
            for recall_index, remembered in enumerate(RECALL_ORDER):
                rows.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "source_label": label,
                        "metric": "within_list_centered_early_lpp",
                        "category": CATEGORY_LABELS[category],
                        "recall_status": RECALL_LABELS[remembered],
                        "mean": float(
                            summary["lpp_mean"][category_index, recall_index]
                        ),
                        "lower": float(
                            summary["lpp_bounds"][0, category_index, recall_index]
                        ),
                        "upper": float(
                            summary["lpp_bounds"][1, category_index, recall_index]
                        ),
                        "interval": interval,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "n_units": n_units,
                    }
                )
        contrast_rows.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "source_label": label,
                "contrast": "Negative minus Neutral recall rate",
                "estimate": float(summary["recall_mean"][0] - summary["recall_mean"][1]),
            }
        )
        for category_index, category in enumerate(CATEGORY_ORDER):
            contrast_rows.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_label": label,
                    "contrast": (
                        f"Remembered minus Forgotten centered Early LPP: "
                        f"{CATEGORY_LABELS[category]}"
                    ),
                    "estimate": float(
                        summary["lpp_mean"][category_index, 0]
                        - summary["lpp_mean"][category_index, 1]
                    ),
                }
            )
    return rows, contrast_rows, summaries


def model_contrast_rows(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["registry_id"]: row for row in comparison}
    specs: list[tuple[str, str, str, str]] = []
    for suffix, lpp in (
        ("", "No"),
        ("_LPP_General", "General"),
        ("_LPP_EmotionalOnly", "Emotion-dependent"),
    ):
        specs.append(
            (
                "Categorical enhancement",
                lpp,
                f"CategoryOnly_eCMR{suffix}",
                f"EEM_eCMR{suffix}",
            )
        )
    for prefix, enhancement in (
        ("CategoryOnly_eCMR", "Absent"),
        ("EEM_eCMR", "Present"),
    ):
        specs.extend(
            [
                ("General LPP versus no LPP", enhancement, prefix, f"{prefix}_LPP_General"),
                (
                    "Emotion-dependent LPP versus no LPP",
                    enhancement,
                    prefix,
                    f"{prefix}_LPP_EmotionalOnly",
                ),
                (
                    "Emotion-dependent versus General LPP",
                    enhancement,
                    f"{prefix}_LPP_General",
                    f"{prefix}_LPP_EmotionalOnly",
                ),
            ]
        )
    rows: list[dict[str, Any]] = []
    for family, stratum, baseline_id, focal_id in specs:
        baseline = by_id[baseline_id]
        focal = by_id[focal_id]
        rows.append(
            {
                "contrast": family,
                "stratum": stratum,
                "baseline_model": baseline["model"],
                "focal_model": focal["model"],
                "NLL_improvement": float(baseline["NLL"]) - float(focal["NLL"]),
                "AIC_improvement": float(baseline["AIC"]) - float(focal["AIC"]),
                "interpretation": "Positive values favor the focal model",
            }
        )
    return rows


def render_diagnostic_figure() -> None:
    subprocess.run(
        ["Rscript", str(FIGURE_SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
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


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.5g}"


def write_model_comparison_qmd(rows: list[dict[str, Any]]) -> None:
    lines = [
        "::: {#tbl-pooled-model-fit}",
        "| Emotion-based learning boost | LPP-based learning boost | $k$ | NLL | AIC | $\\Delta$AIC |",
        "|:--:|:--|--:|--:|--:|--:|",
    ]
    for row in rows:
        best = math.isclose(float(row["delta_AIC"]), 0.0, abs_tol=1e-12)
        nll = f'{float(row["NLL"]):.3f}'
        aic = f'{float(row["AIC"]):.3f}'
        delta = f'{float(row["delta_AIC"]):.3f}'
        if best:
            nll = f"**{nll}**"
            aic = f"**{aic}**"
            delta = f"**{delta}**"
        lines.append(
            "| {boost} | {lpp_scope} | {k} | {nll} | {aic} | {delta} |".format(
                boost=EMOTION_BOOST_LABELS[row["categorical_enhancement"]],
                lpp_scope=LPP_BOOST_LABELS[row["lpp_modulation"]],
                k=row["free_parameters"],
                nll=nll,
                aic=aic,
                delta=delta,
            )
        )
    lines.extend(
        [
            "",
            "Pooled comparison of six models that share the same eCMR architecture. An emotion-based learning boost is one multiplier shared by all negative items; an LPP-based learning boost varies with each item's Early LPP and is fitted either for all items or for negative items only. The table indicates which boosts were available for estimation, not whether they improved fit or reproduced a particular empirical pattern. $k$ is the number of free parameters; NLL is the negative log-likelihood for one parameter vector fitted to all 342 lists from 38 participants; $\\Delta$AIC is relative to the best model. Bold values identify the best-fitting model. Every fitted Early-LPP coefficient was constrained to $[0, 0.2145443]$. The comparison between the all-item and negative-item-only LPP-based boosts in models with an emotion-based boost is conditional on this range because the fitted all-item coefficient ($\\hat{\\kappa}=0.20799$) approached its upper bound and no wider-bound sensitivity analysis was performed.",
            ":::" ,
            "",
        ]
    )
    (PACKAGE_DIR / "model_comparison_table.qmd").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def load_original_lpp_contrasts() -> list[dict[str, Any]]:
    if not ORIGINAL_LPP_CONTRAST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ORIGINAL_LPP_CONTRAST_PATH}. Run "
            "work/lpp_model_prediction_grids/"
            "build_selected_early_lpp_figure.py first."
        )
    with ORIGINAL_LPP_CONTRAST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 14:
        raise ValueError(
            f"Expected 14 original-scale Early-LPP contrasts, found {len(rows)}"
        )
    return rows


def write_diagnostic_effects_qmd(
    recall_rows: list[dict[str, Any]],
    lpp_rows: list[dict[str, Any]],
) -> None:
    contrast_names = (
        "Negative minus Neutral recall rate",
        "Remembered minus Forgotten Early LPP: Negative",
        "Remembered minus Forgotten Early LPP: Neutral",
    )
    expected_sources = {"Observed"} | set(MODEL_BY_ID)
    lookup: dict[tuple[str, str], float] = {}
    rows = [
        row
        for row in recall_rows
        if row["contrast"] == "Negative minus Neutral recall rate"
    ] + lpp_rows
    for row in rows:
        source_id = str(row["source_id"])
        contrast = str(row["contrast"])
        key = (source_id, contrast)
        if source_id not in expected_sources:
            raise ValueError(f"Unexpected diagnostic source: {source_id}")
        if contrast not in contrast_names:
            raise ValueError(f"Unexpected diagnostic contrast: {contrast}")
        if key in lookup:
            raise ValueError(f"Duplicate diagnostic contrast: {key}")
        lookup[key] = float(row["estimate"])

    expected_keys = {
        (source_id, contrast)
        for source_id in expected_sources
        for contrast in contrast_names
    }
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))
        extra = sorted(set(lookup) - expected_keys)
        raise ValueError(
            f"Diagnostic contrast table mismatch; missing={missing}, extra={extra}"
        )

    def values(source_id: str) -> tuple[float, float, float]:
        return tuple(lookup[(source_id, contrast)] for contrast in contrast_names)

    def format_discrepancy(estimate: float, target: float) -> str:
        discrepancy = estimate - target
        return f"{discrepancy:+.3f}".replace("-", "−")

    observed = values("Observed")
    lines = [
        "::: {#tbl-pooled-model-diagnostic-effects}",
        "| eCMR specification | Recall rate: Negative − Neutral | Early LPP: Remembered − Forgotten, Negative items | Early LPP: Remembered − Forgotten, Neutral items |",
        "|:--|--:|--:|--:|",
        (
            "| **Observed data** | "
            f"{observed[0]:.3f} | {observed[1]:.3f} | {observed[2]:.3f} |"
        ),
    ]
    for model in MODELS:
        estimates = values(model.registry_id)
        lines.append(
            "| {model} | {recall:.3f} *({recall_error})* | "
            "{negative_lpp:.3f} *({negative_lpp_error})* | "
            "{neutral_lpp:.3f} *({neutral_lpp_error})* |".format(
                model=MODEL_SPECIFICATION_LABELS[model.registry_id],
                recall=estimates[0],
                recall_error=format_discrepancy(estimates[0], observed[0]),
                negative_lpp=estimates[1],
                negative_lpp_error=format_discrepancy(
                    estimates[1], observed[1]
                ),
                neutral_lpp=estimates[2],
                neutral_lpp_error=format_discrepancy(
                    estimates[2], observed[2]
                ),
            )
        )
    lines.extend(
        [
            "",
            "Observed and model-predicted contrasts in recall rate and Early LPP. All model rows share the same eCMR architecture. An emotion-based learning boost is one multiplier shared by all negative items; an LPP-based learning boost varies with each item's Early LPP. The model names indicate which boosts were available for estimation, not whether they reproduced a particular empirical pattern. Each main numerical entry is the first-named condition minus the second, so positive values indicate greater recall for negative than neutral items or greater Early LPP for remembered than forgotten items. For every prediction, the italicized value in parentheses is prediction minus observed; values nearer zero indicate closer reproduction of the observed contrast. Predictions are means across 200 complete simulated datasets. The Early-LPP columns use the original z-transformed values reported in the empirical analysis: values are averaged within participant and cell, then across participants. The fitted models used within-list-centered Early LPP as a predictor, but the displayed diagnostic is the original-scale EEG outcome. The models were fitted to recalled-item identities rather than to these contrast summaries, so the Early-LPP contrasts are derived diagnostics rather than separately fitted outcomes.",
            ":::" ,
            "",
        ]
    )
    (PACKAGE_DIR / "diagnostic_effects_table.qmd").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_parameter_qmd(
    parameter_data: list[dict[str, Any]],
) -> None:
    lookup = {
        (row["implementation_parameter"], row["model_code"]): row
        for row in parameter_data
    }
    expected_parameter_names = {name for name, _symbol, _description in PARAMETERS}
    if set(PARAMETER_RANGES) != expected_parameter_names:
        raise ValueError("Parameter-range metadata do not match PARAMETERS")
    expected_cell_count = len(PARAMETERS) * len(MODELS)
    if len(lookup) != expected_cell_count or len(parameter_data) != expected_cell_count:
        raise ValueError(
            f"Expected {expected_cell_count} unique parameter cells, "
            f"found {len(lookup)} unique and {len(parameter_data)} total"
        )

    lines = [
        "**Model-column key**",
        "",
        "| Emotion-based learning boost | LPP-based boost: None | LPP-based boost: All items | LPP-based boost: Negative items only |",
        "|:--|:--:|:--:|:--:|",
        "| No | A | B | C |",
        "| Yes | D | E | F |",
    ]
    lines.extend(
        [
            "",
            "| Parameter | Role | Range / fixed value | A | B | C | D | E | F |",
            "|:--|:--|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for implementation_name, symbol, description in PARAMETERS:
        values = []
        for model in MODELS:
            row = lookup[(implementation_name, model.short_code)]
            formatted = _format_number(float(row["estimate"]))
            values.append(f"[{formatted}]" if row["status"] == "fixed" else formatted)
        lines.append(
            f"| {symbol} | {description} | {PARAMETER_RANGES[implementation_name]} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Pooled best-found parameter estimates for the six primary source-context models, rounded to five significant digits. Values in square brackets were fixed rather than estimated. Bounds shown as open at zero were implemented with machine epsilon as the positive lower bound. The complete-precision estimates and implementation parameter names are provided in `parameter_estimates.csv`. {#tbl-model-parameters}",
            "",
        ]
    )
    (PACKAGE_DIR / "parameter_table.qmd").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_figure_qmd() -> None:
    alt = (
        "Five rows of paired horizontal grouped-bar charts show observed data followed by "
        "four selected model predictions. The left column compares Negative and "
        "Neutral recall rates. Emotion-dependent LPP without categorical "
        "enhancement does not reproduce the observed Negative advantage, whereas "
        "the three categorical-enhancement models do. The right column compares "
        "Remembered and Forgotten items' mean within-list-centered Early LPP in "
        "each category. Categorical enhancement without LPP shows little "
        "separation; General LPP separates both categories to differing degrees; "
        "and Emotion-dependent LPP concentrates the separation in Negative items."
    )
    caption = (
        "Observed benchmarks and four predictions selected from the six pooled "
        "source-context models to illustrate the planned mechanistic contrasts. "
        "Bars in the left column are overall recall rates for Negative and Neutral "
        "items. Bars in the right column are mean within-list-centered Early LPP "
        "values for Remembered and Forgotten items in each category. Comparing "
        "the first and fourth prediction rows isolates categorical enhancement; "
        "comparing the second and fourth isolates Emotion-dependent LPP; and "
        "comparing the third and fourth contrasts equal-complexity General and "
        "Emotion-dependent LPP mappings. All six fits are reported in the model-"
        "comparison and parameter tables. "
        "Observed error bars are percentile 95% confidence intervals from 10,000 "
        "participant-cluster bootstrap samples ($N=38$). Predicted error bars are "
        "central 95% intervals across 200 complete simulated datasets and therefore "
        "represent simulation variability, not parameter-estimation uncertainty. "
        "Simulations condition on each list's observed recall count. The LPP-by-"
        "subsequent-memory contrast is a derived diagnostic from the fitted "
        "recalled-set model, not an independently optimized target or an out-of-"
        "sample validation."
    )
    fragment = (
        f'![{caption}](diagnostic_figure.svg){{#fig-model-diagnostics '
        f'width="100%" fig-alt="{alt}"}}\n'
    )
    (PACKAGE_DIR / "diagnostic_figure_caption.qmd").write_text(
        fragment, encoding="utf-8"
    )
    (PACKAGE_DIR / "diagnostic_figure_alt.txt").write_text(
        alt + "\n", encoding="utf-8"
    )


def write_results_readout(
    comparison: list[dict[str, Any]],
    model_contrasts: list[dict[str, Any]],
    diagnostic_contrasts: list[dict[str, Any]],
    original_lpp_contrasts: list[dict[str, Any]],
) -> None:
    by_id = {row["registry_id"]: row for row in comparison}
    model_contrast = {
        (row["contrast"], row["stratum"]): row for row in model_contrasts
    }
    diagnostic = {
        (row["source_id"], row["contrast"]): row
        for row in diagnostic_contrasts
        if row["contrast"] == "Negative minus Neutral recall rate"
    }
    diagnostic.update(
        {
            (row["source_id"], row["contrast"]): row
            for row in original_lpp_contrasts
        }
    )
    winner = min(comparison, key=lambda row: float(row["AIC"]))
    lines = [
        "## Six-model results readout",
        "",
        f"The best of the selected fits was **{winner['model']}** "
        f"(NLL = {winner['NLL']:.3f}, AIC = {winner['AIC']:.3f}).",
        "",
        "Across all three LPP specifications, categorical enhancement improved AIC:",
        "",
    ]
    for stratum in ("No", "General", "Emotion-dependent"):
        gain = model_contrast[("Categorical enhancement", stratum)]["AIC_improvement"]
        lines.append(f"- {stratum} LPP: $\\Delta$AIC improvement = {gain:.3f}.")
    lines.extend(
        [
            "",
            "At equal complexity, Emotion-dependent LPP was preferred to General LPP in both categorical-enhancement strata:",
            "",
        ]
    )
    for stratum in ("Absent", "Present"):
        gain = model_contrast[("Emotion-dependent versus General LPP", stratum)][
            "AIC_improvement"
        ]
        lines.append(
            f"- Categorical enhancement {stratum.lower()}: $\\Delta$AIC improvement = {gain:.3f}."
        )
    observed_recall_gap = diagnostic[("Observed", "Negative minus Neutral recall rate")][
        "estimate"
    ]
    combined_id = "EEM_eCMR_LPP_EmotionalOnly"
    combined_recall_gap = diagnostic[
        (combined_id, "Negative minus Neutral recall rate")
    ]["estimate"]
    observed_negative_lpp = diagnostic[
        ("Observed", "Remembered minus Forgotten Early LPP: Negative")
    ]["estimate"]
    observed_neutral_lpp = diagnostic[
        ("Observed", "Remembered minus Forgotten Early LPP: Neutral")
    ]["estimate"]
    combined_negative_lpp = diagnostic[
        (combined_id, "Remembered minus Forgotten Early LPP: Negative")
    ]["estimate"]
    combined_neutral_lpp = diagnostic[
        (combined_id, "Remembered minus Forgotten Early LPP: Neutral")
    ]["estimate"]
    (
        observed_recall_gap,
        combined_recall_gap,
        observed_negative_lpp,
        observed_neutral_lpp,
        combined_negative_lpp,
        combined_neutral_lpp,
    ) = map(
        float,
        (
            observed_recall_gap,
            combined_recall_gap,
            observed_negative_lpp,
            observed_neutral_lpp,
            combined_negative_lpp,
            combined_neutral_lpp,
        ),
    )
    lines.extend(
        [
            "",
            "The grouped-bar diagnostics make the proposed division of explanatory work visible. The observed Negative-minus-Neutral recall-rate difference was "
            f"{observed_recall_gap:.3f}; the categorical-enhancement plus Emotion-dependent LPP model predicted {combined_recall_gap:.3f}. "
            "The observed Remembered-minus-Forgotten original-z Early-LPP differences were "
            f"{observed_negative_lpp:.3f} for Negative items and {observed_neutral_lpp:.3f} for Neutral items; the combined model predicted "
            f"{combined_negative_lpp:.3f} and {combined_neutral_lpp:.3f}, respectively.",
            "",
            "The General-versus-Emotion-dependent conclusion with categorical enhancement is conditional on the declared nonnegative LPP-coefficient range $[0, 0.2145443]$: `EEM_eCMR_LPP_General` fitted $\\hat{\\kappa}=0.20799$, near the upper bound, and no wider-bound sensitivity analysis was performed. These results do not establish that a residual General component is zero because Full-LPP models are outside this manuscript comparison.",
            "",
            "The LPP-by-memory-status bars are derived diagnostics. The models were fitted to recalled sets and the simulations condition on observed list recall counts; the bars are not independent validation and do not test recall termination or total recall.",
            "",
        ]
    )
    (PACKAGE_DIR / "results_readout.qmd").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(source_paths: list[Path], generated_paths: list[Path]) -> None:
    manifest = {
        "package": "work/lpp_model_results",
        "purpose": "Issue #10 six-model pooled tables and grouped-bar diagnostics",
        "model_ids": [model.registry_id for model in MODELS],
        "figure_model_ids": list(FIGURE_MODEL_IDS),
        "figure_selection_rationale": (
            "Four predictions isolate categorical enhancement, the addition of "
            "LPP, and General versus Emotion-dependent LPP while all six fitted "
            "models remain in the numerical tables"
        ),
        "source_artifacts": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
        "generated_artifacts": {
            path.name: _sha256(path) for path in generated_paths
        },
        "summary_settings": {
            "observed_point_estimate": "item-pooled within each displayed cell",
            "observed_resampling_unit": "participant",
            "observed_units": 38,
            "observed_interval": "percentile participant-cluster bootstrap",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "simulation_unit": "complete simulated dataset across all 342 lists",
            "simulation_units_per_model": 200,
            "simulation_interval": "central interval across simulated datasets",
            "confidence_level": CONFIDENCE_LEVEL,
            "lpp_preprocessing": "within-list mean centering of EarlyLPP",
            "manuscript_lpp_contrast_source": (
                "original-z participant-mean contrasts from "
                "work/lpp_model_prediction_grids/original_early_lpp_contrasts.csv"
            ),
            "recall_status": "presentation position appears at least once in recalls",
        },
        "declared_lpp_bounds": [0.0, 0.21454430108484696],
        "scope_exclusions": [
            "temporal-only models",
            "Full-LPP models",
            "subject-wise process-model fits",
            "wider-bound sensitivity analysis",
        ],
        "software": {
            "python": __import__("sys").version.split()[0],
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
    path = PACKAGE_DIR / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(FIGURE_MODEL_IDS) != 4 or len(set(FIGURE_MODEL_IDS)) != 4:
        raise ValueError("Diagnostic figure requires four unique model IDs")
    unknown_figure_models = set(FIGURE_MODEL_IDS) - set(MODEL_BY_ID)
    if unknown_figure_models:
        raise ValueError(
            "Unknown diagnostic-figure model IDs: "
            + ", ".join(sorted(unknown_figure_models))
        )

    run_manifest = _load_json(RUN_MANIFEST_PATH)
    if run_manifest["fit_settings"]["pooled"] is not True:
        raise ValueError("Issue #10 package requires pooled fits")
    if run_manifest["fit_settings"]["lpp_preprocessing"] != (
        "within-list mean centering of EarlyLPP"
    ):
        raise ValueError("Unexpected LPP preprocessing in fit manifest")

    comparison, fit_results = load_fit_results()
    parameters = parameter_rows(fit_results)
    diagnostics, diagnostic_contrasts, _summaries = diagnostic_rows(run_manifest)
    planned_contrasts = model_contrast_rows(comparison)
    original_lpp_contrasts = load_original_lpp_contrasts()

    _write_csv(
        PACKAGE_DIR / "model_comparison.csv",
        comparison,
        (
            "model_code",
            "model",
            "registry_id",
            "categorical_enhancement",
            "lpp_modulation",
            "free_parameters",
            "NLL",
            "AIC",
            "delta_AIC",
            "Akaike_weight",
        ),
    )
    _write_csv(
        PACKAGE_DIR / "parameter_estimates.csv",
        parameters,
        (
            "model_code",
            "model",
            "registry_id",
            "implementation_parameter",
            "manuscript_symbol",
            "description",
            "estimate",
            "status",
            "lower_bound",
            "upper_bound",
        ),
    )
    _write_csv(
        PACKAGE_DIR / "diagnostic_summary.csv",
        diagnostics,
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
        PACKAGE_DIR / "diagnostic_contrasts.csv",
        diagnostic_contrasts,
        ("source_type", "source_id", "source_label", "contrast", "estimate"),
    )
    _write_csv(
        PACKAGE_DIR / "planned_model_contrasts.csv",
        planned_contrasts,
        (
            "contrast",
            "stratum",
            "baseline_model",
            "focal_model",
            "NLL_improvement",
            "AIC_improvement",
            "interpretation",
        ),
    )

    write_model_comparison_qmd(comparison)
    write_diagnostic_effects_qmd(diagnostic_contrasts, original_lpp_contrasts)
    write_parameter_qmd(parameters)
    write_figure_qmd()
    write_results_readout(
        comparison,
        planned_contrasts,
        diagnostic_contrasts,
        original_lpp_contrasts,
    )
    render_diagnostic_figure()

    generated = [
        PACKAGE_DIR / name
        for name in (
            "model_comparison.csv",
            "model_comparison_table.qmd",
            "diagnostic_effects_table.qmd",
            "parameter_estimates.csv",
            "parameter_table.qmd",
            "diagnostic_summary.csv",
            "diagnostic_contrasts.csv",
            "planned_model_contrasts.csv",
            "diagnostic_figure.svg",
            "diagnostic_figure.pdf",
            "diagnostic_figure.png",
            "diagnostic_figure_caption.qmd",
            "diagnostic_figure_alt.txt",
            "results_readout.qmd",
        )
    ]
    source_paths = [
        DATA_PATH,
        RUN_MANIFEST_PATH,
        Path(__file__),
        FIGURE_SCRIPT_PATH,
        ORIGINAL_LPP_CONTRAST_PATH,
    ]
    for model in MODELS:
        source_paths.extend(_fit_paths(model))
    write_manifest(source_paths, generated)

    print(
        json.dumps(
            {
                "package": PACKAGE_DIR.relative_to(PROJECT_ROOT).as_posix(),
                "models": len(MODELS),
                "generated": len(generated) + 1,
                "best_model": min(comparison, key=lambda row: row["AIC"])["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
