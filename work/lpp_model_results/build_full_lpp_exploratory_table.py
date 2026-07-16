#!/usr/bin/env python3
"""Build the exploratory source-context Full-LPP supplementary table."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp

from jaxcmr.helpers import generate_trial_mask, import_from_string, load_data

from build_results import PARAMETERS, PARAMETER_RANGES


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
FIT_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"
LOSS_FN_PATH = (
    "jaxcmr.loss.set_permutation_likelihood.ExcludeTerminationLikelihoodLoss"
)
MAKE_FACTORY_PATH = "lpp_ecmr.models.full_eeg_ecmr.make_factory"
COMPONENT_PATHS = {
    "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
    "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
    "context_create_fn": "jaxcmr.components.context.init",
    "termination_policy_create_fn": (
        "jaxcmr.components.termination.NoStopTermination"
    ),
}
TRIAL_QUERY = "data['subject'] > -1"
NLL_TOLERANCE = 1e-4


@dataclass(frozen=True)
class FullSpec:
    full_id: str
    parent_id: str
    emotion_boost: str
    raw_code: str


FULL_SPECS = (
    FullSpec(
        "CategoryOnly_eCMR_LPP_Full",
        "CategoryOnly_eCMR_LPP_EmotionalOnly",
        "No",
        "G",
    ),
    FullSpec(
        "EEM_eCMR_LPP_Full",
        "EEM_eCMR_LPP_EmotionalOnly",
        "Yes",
        "H",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(value: Any) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one value, got {value!r}")
        value = value[0]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {result}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: Iterable[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _fit_path(model_id: str) -> Path:
    return FIT_DIR / f"{model_id}_best_of_3.json"


def _fit_parameters(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: _scalar(value)
        for name, value in result["fits"].items()
        if name != "subject"
    }


def _validate_fit(result: dict[str, Any], expected_id: str) -> None:
    if result.get("model") != expected_id:
        raise ValueError(
            f"Expected {expected_id}, found {result.get('model')!r}"
        )
    if result.get("loss_function") != LOSS_FN_PATH:
        raise ValueError(f"Unexpected loss for {expected_id}")
    if result.get("model_factory") != MAKE_FACTORY_PATH:
        raise ValueError(f"Unexpected model factory for {expected_id}")
    if result.get("trial_query") != TRIAL_QUERY:
        raise ValueError(f"Unexpected trial query for {expected_id}")
    if not bool(result.get("converged", [False])[0]):
        raise ValueError(f"Optimizer did not report convergence for {expected_id}")

    estimates = _fit_parameters(result)
    for name, bounds in result["free"].items():
        lower, upper = map(float, bounds)
        estimate = estimates[name]
        if not lower <= estimate <= upper:
            raise ValueError(
                f"{expected_id} {name}={estimate} outside {bounds}"
            )


def _make_loss() -> tuple[Any, jax.Array]:
    data = load_data(DATA_PATH, 0)
    trial_indices = jnp.where(generate_trial_mask(data, TRIAL_QUERY))[0]
    if int(trial_indices.size) != 342:
        raise ValueError(f"Expected 342 lists, found {trial_indices.size}")

    make_factory = import_from_string(MAKE_FACTORY_PATH)
    factory = make_factory(
        **{
            key: import_from_string(path)
            for key, path in COMPONENT_PATHS.items()
        }
    )
    loss_cls = import_from_string(LOSS_FN_PATH)
    return loss_cls(factory, data, None), trial_indices


def _evaluate_candidates(
    loss: Any,
    trial_indices: jax.Array,
    full: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    free_names = tuple(full["free"])
    raw_params = _fit_parameters(full)
    parent_params = _fit_parameters(parent)
    embedded_params = {
        name: 0.0 if name == "lpp_main_scale" else parent_params[name]
        for name in free_names
    }

    for name, estimate in embedded_params.items():
        lower, upper = map(float, full["free"][name])
        if not lower <= estimate <= upper:
            raise ValueError(
                f"Embedded candidate {name}={estimate} outside "
                f"Full bounds {full['free'][name]}"
            )

    population = jnp.asarray(
        [
            [raw_params[name] for name in free_names],
            [embedded_params[name] for name in free_names],
        ],
        dtype=jnp.float32,
    ).T
    evaluate = jax.jit(
        lambda indices, values: loss(
            indices, full["fixed"], free_names, values
        )
    )
    raw_nll, embedded_nll = map(
        float, jax.device_get(evaluate(trial_indices, population)).tolist()
    )

    stored_raw_nll = _scalar(full["fitness"])
    stored_parent_nll = _scalar(parent["fitness"])
    if abs(raw_nll - stored_raw_nll) > NLL_TOLERANCE:
        raise AssertionError(
            f"Raw Full reevaluation differs from stored NLL: "
            f"{raw_nll} versus {stored_raw_nll}"
        )
    if abs(embedded_nll - stored_parent_nll) > NLL_TOLERANCE:
        raise AssertionError(
            f"Embedded-parent NLL differs from parent NLL: "
            f"{embedded_nll} versus {stored_parent_nll}"
        )
    return raw_nll, embedded_nll, raw_params, embedded_params


def _display_value(estimate: float, status: str) -> str:
    value = f"{estimate:.5g}"
    return f"[{value}]" if status == "fixed" else value


def _write_qmd(
    audit_rows: list[dict[str, Any]], parameter_rows: list[dict[str, Any]]
) -> None:
    by_code_parameter = {
        (row["model_code"], row["implementation_parameter"]): row
        for row in parameter_rows
    }
    lines = [
        "::: {#tbl-full-lpp-exploratory}",
        "**Panel A. Original Full-model optimizer results**",
        "",
        "| Emotion-based learning boost | $k$ | Full NLL | Full AIC | NLL above the $\\kappa=0$ model |",
        "|:--|--:|--:|--:|--:|",
    ]
    for row in audit_rows:
        lines.append(
            "| {emotion_based_learning_boost} | {full_parameter_count} | "
            "{raw_full_nll:.3f} | {raw_full_aic:.3f} | "
            "{raw_nll_minus_parent:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "**Panel B. Original Full-model parameter estimates**",
            "",
            "**Model-column key**",
            "",
            "| Model column | Emotion-based learning boost |",
            "|:--:|:--:|",
        ]
    )
    for spec in FULL_SPECS:
        lines.append(f"| {spec.raw_code} | {spec.emotion_boost} |")

    lines.extend(
        [
            "",
            "| Parameter | Role | Range / fixed value | G | H |",
            "|:--|:--|:--|--:|--:|",
        ]
    )
    for implementation_name, symbol, description in PARAMETERS:
        if implementation_name == "lpp_inter_scale":
            description = (
                "Additional negative-item Early-LPP slope on log learning strength"
            )
        values = [
            _display_value(
                float(by_code_parameter[(code, implementation_name)]["estimate"]),
                str(by_code_parameter[(code, implementation_name)]["status"]),
            )
            for code in ("G", "H")
        ]
        lines.append(
            f"| {symbol} | {description} | "
            f"{PARAMETER_RANGES[implementation_name]} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "Exploratory source-context Full-LPP results. The Full mapping estimates both the general Early-LPP slope, $\\kappa$, and the additional negative-item slope, $\\kappa_{\\mathrm{emot}}$. G and H are the original best-of-three optimizer outputs without and with the emotion-based learning boost, respectively. Values in square brackets were fixed by the model.",
            "",
            "Both original Full solutions had higher NLLs than the corresponding $\\kappa=0$ models already reported in the primary comparison. Evaluating those reported parameter vectors through the Full-model likelihood at $\\kappa=0$ reproduced their NLLs exactly at the stored precision. Their NLLs and parameter vectors are therefore not repeated here. The original Full outputs are retained as exploratory optimization attempts rather than maximum-likelihood estimates, and they cannot establish that the residual general Early-LPP slope is zero. Complete-precision Full estimates are provided in `full_lpp_exploratory_parameters.csv`; the boundary evaluations and source-file hashes are retained in `full_lpp_embedded_parent_evaluations.json`.",
            ":::",
            "",
        ]
    )
    (PACKAGE_DIR / "full_lpp_exploratory_table.qmd").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    loss, trial_indices = _make_loss()
    audit_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    audit_models: list[dict[str, Any]] = []

    for spec in FULL_SPECS:
        full_path = _fit_path(spec.full_id)
        parent_path = _fit_path(spec.parent_id)
        full = _load_json(full_path)
        parent = _load_json(parent_path)
        _validate_fit(full, spec.full_id)
        _validate_fit(parent, spec.parent_id)

        raw_nll, embedded_nll, raw_params, embedded_free_params = (
            _evaluate_candidates(loss, trial_indices, full, parent)
        )
        parent_nll = _scalar(parent["fitness"])
        parameter_count = len(full["free"])
        raw_aic = 2 * raw_nll + 2 * parameter_count
        audit_rows.append(
            {
                "emotion_based_learning_boost": spec.emotion_boost,
                "full_registry_id": spec.full_id,
                "nested_parent_registry_id": spec.parent_id,
                "raw_full_nll": raw_nll,
                "raw_full_aic": raw_aic,
                "nested_parent_nll": parent_nll,
                "raw_nll_minus_parent": raw_nll - parent_nll,
                "embedded_parent_nll": embedded_nll,
                "embedded_minus_parent": embedded_nll - parent_nll,
                "full_parameter_count": parameter_count,
                "optimization_status": (
                    "Raw optimizer output violates nesting; "
                    "embedded parent verified"
                ),
            }
        )

        full_fixed = {
            name: _scalar(value) for name, value in full["fixed"].items()
        }
        embedded_params = {**full_fixed, **embedded_free_params}
        for implementation_name, symbol, description in PARAMETERS:
            if implementation_name not in raw_params:
                raise KeyError(f"{spec.full_id} lacks {implementation_name}")
            if implementation_name in full["fixed"]:
                status = "fixed"
                provenance = "fixed by Full model"
            else:
                status = "free"
                provenance = "raw Full optimizer output"
            bounds = full["free"].get(implementation_name)
            parameter_rows.append(
                {
                    "model_code": spec.raw_code,
                    "emotion_based_learning_boost": spec.emotion_boost,
                    "full_registry_id": spec.full_id,
                    "implementation_parameter": implementation_name,
                    "manuscript_symbol": symbol.replace("$", ""),
                    "description": description,
                    "estimate": raw_params[implementation_name],
                    "status": status,
                    "provenance": provenance,
                    "lower_bound": "" if bounds is None else float(bounds[0]),
                    "upper_bound": "" if bounds is None else float(bounds[1]),
                }
            )

        audit_models.append(
            {
                "emotion_based_learning_boost": spec.emotion_boost,
                "full_model": spec.full_id,
                "nested_parent_model": spec.parent_id,
                "source_files": {
                    "full": {
                        "path": str(full_path.relative_to(PROJECT_ROOT)),
                        "sha256": _sha256(full_path),
                    },
                    "parent": {
                        "path": str(parent_path.relative_to(PROJECT_ROOT)),
                        "sha256": _sha256(parent_path),
                    },
                },
                "raw_full": {
                    "stored_nll": _scalar(full["fitness"]),
                    "reevaluated_nll": raw_nll,
                    "aic": raw_aic,
                    "free_parameter_count": parameter_count,
                    "parameters": raw_params,
                },
                "nested_parent": {
                    "stored_nll": parent_nll,
                    "free_parameter_count": len(parent["free"]),
                },
                "embedded_parent_candidate": {
                    "evaluated_full_model_nll": embedded_nll,
                    "difference_from_parent_nll": embedded_nll - parent_nll,
                    "aic_counting_full_parameters": (
                        2 * embedded_nll + 2 * parameter_count
                    ),
                    "parameters": embedded_params,
                    "general_lpp_slope_status": (
                        "free in Full model; evaluated at boundary zero"
                    ),
                },
                "nesting_result": (
                    "Embedded parent has lower NLL than raw Full optimizer output"
                ),
            }
        )

    if len(audit_rows) != 2 or len(parameter_rows) != 30:
        raise AssertionError(
            f"Expected 2 audit rows and 30 parameter rows; found "
            f"{len(audit_rows)} and {len(parameter_rows)}"
        )
    if any(abs(row["embedded_minus_parent"]) > NLL_TOLERANCE for row in audit_rows):
        raise AssertionError("An embedded-parent evaluation did not match its parent")

    _write_csv(
        PACKAGE_DIR / "full_lpp_exploratory_fits.csv",
        audit_rows,
        audit_rows[0].keys(),
    )
    _write_csv(
        PACKAGE_DIR / "full_lpp_exploratory_parameters.csv",
        parameter_rows,
        parameter_rows[0].keys(),
    )
    audit = {
        "analysis": "Exploratory source-context Full-LPP nesting audit",
        "scope": (
            "Two source-context Full variants; temporal-only Full variants excluded"
        ),
        "evaluation_check": (
            "Each reported kappa=0 parent was explicitly evaluated through the "
            "corresponding Full-model likelihood"
        ),
        "likelihood": LOSS_FN_PATH,
        "model_factory": MAKE_FACTORY_PATH,
        "components": COMPONENT_PATHS,
        "trial_query": TRIAL_QUERY,
        "list_count": int(trial_indices.size),
        "nll_tolerance": NLL_TOLERANCE,
        "data": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(DATA_PATH),
        },
        "models": audit_models,
    }
    (PACKAGE_DIR / "full_lpp_embedded_parent_evaluations.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    _write_qmd(audit_rows, parameter_rows)


if __name__ == "__main__":
    main()
