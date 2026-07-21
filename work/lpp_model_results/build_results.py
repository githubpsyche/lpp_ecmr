#!/usr/bin/env python3
"""Build the six-model result tables and diagnostic summaries.

The script reads the returned pooled fit JSON and simulation HDF5 artifacts from
``work/lpp_model_comparison``.  It does not copy or modify those source files.
All products are written beside this script so that the result package remains
flat, bounded, and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from lpp_ecmr.data_contract import (
    MIXED_EXPECTED_LISTS,
    MIXED_EXPECTED_SUBJECTS,
    MIXED_TRIAL_QUERY,
    mixed_trial_mask,
    slice_trials,
)
from lpp_ecmr.model_comparison_registry import FIT_SETTINGS


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
FIT_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"

BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
EXPECTED_SIMULATIONS = int(FIT_SETTINGS["experiment_count"])

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
        if result.get("model") != model.registry_id:
            raise ValueError(
                f"Expected {model.registry_id}, found {result.get('model')!r}"
            )
        if result.get("trial_query") != MIXED_TRIAL_QUERY:
            raise ValueError(
                f"{model.registry_id} was not fitted with {MIXED_TRIAL_QUERY!r}"
            )
        fit_subjects = np.asarray(result.get("fits", {}).get("subject", []))
        if fit_subjects.size != 1 or int(fit_subjects.reshape(-1)[0]) != -1:
            raise ValueError(f"Selected fit is not pooled: {model.registry_id}")
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


def _load_recall_data(path: Path, *, mixed_only: bool = False) -> dict[str, np.ndarray]:
    keys = (
        "subject",
        "list",
        "listLength",
        "condition",
        "pres_itemnos",
        "recalls",
        "EarlyLPP",
    )
    if mixed_only:
        keys += ("list_type",)
    with h5py.File(path, "r") as handle:
        group = handle["data"]
        data = {key: np.asarray(group[key][:]) for key in keys}

    if mixed_only:
        trial_major = {key: value.T for key, value in data.items()}
        trial_major = slice_trials(
            trial_major,
            mixed_trial_mask(trial_major),
        )
        data = {key: np.asarray(value).T for key, value in trial_major.items()}

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


def diagnostic_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
]:
    expected_lists = MIXED_EXPECTED_LISTS
    expected_simulations = EXPECTED_SIMULATIONS
    observed = _load_recall_data(DATA_PATH, mixed_only=True)
    (
        subject_ids,
        observed_recall_numerators,
        observed_recall_denominators,
        observed_lpp_sums,
        observed_lpp_counts,
    ) = summarize_observed(observed)
    if observed_recall_numerators.shape[0] != MIXED_EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {MIXED_EXPECTED_SUBJECTS} participant summaries")

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


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.5g}"


def write_model_comparison_qmd(
    rows: list[dict[str, Any]],
    parameter_data: list[dict[str, Any]],
) -> None:
    general_lpp = next(
        float(row["estimate"])
        for row in parameter_data
        if row["registry_id"] == "EEM_eCMR_LPP_General"
        and row["implementation_parameter"] == "lpp_main_scale"
    )
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
            f"Pooled comparison of six models that share the same eCMR architecture. An emotion-based learning boost is one multiplier shared by all negative items; an LPP-based learning boost varies with each item's Early LPP and is fitted either for all items or for negative items only. The table indicates which boosts were available for estimation, not whether they improved fit or reproduced a particular empirical pattern. $k$ is the number of free parameters; NLL is the negative log-likelihood for one parameter vector fitted to all {MIXED_EXPECTED_LISTS} mixed lists from {MIXED_EXPECTED_SUBJECTS} participants; $\\Delta$AIC is relative to the best model. Bold values identify the best-fitting model. Every fitted Early-LPP coefficient was constrained to $[0, 0.2145443]$; the fitted all-item coefficient in the model with an emotion-based learning boost was $\\hat{{\\kappa}}={general_lpp:.5f}$.",
            ":::" ,
            "",
        ]
    )
    (PACKAGE_DIR / "model_comparison_table.qmd").write_text(
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


def main(*, write_prose: bool = True) -> None:
    comparison, fit_results = load_fit_results()
    parameters = parameter_rows(fit_results)
    diagnostics, diagnostic_contrasts, _summaries = diagnostic_rows()
    planned_contrasts = model_contrast_rows(comparison)

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

    if write_prose:
        write_model_comparison_qmd(comparison, parameters)
        write_parameter_qmd(parameters)

    print(
        json.dumps(
            {
                "package": PACKAGE_DIR.relative_to(PROJECT_ROOT).as_posix(),
                "models": len(MODELS),
                "best_model": min(comparison, key=lambda row: row["AIC"])["model"],
                "write_prose": write_prose,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-prose",
        action="store_true",
        help="Refresh numerical and figure artifacts without rewriting QMD/TXT prose.",
    )
    args = parser.parse_args()
    main(write_prose=not args.no_prose)
