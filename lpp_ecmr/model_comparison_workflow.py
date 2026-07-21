"""Prepare and review the flat pooled model-comparison work unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from jaxcmr.helpers import import_from_string, load_data

from .data_contract import mixed_trial_mask, slice_trials
from .model_comparison_registry import (
    EXPECTED_MODEL_NAMES,
    FIT_SETTINGS,
    LPP_CENTERED_MAX,
    LPP_CENTERED_MIN,
    LPP_SLOPE_BOUNDS,
    MODEL_COMPARISON_REGISTRY,
    NESTING_RELATIONSHIPS,
    PARAMETER_MANUSCRIPT_NAMES,
    PARAMETER_NEUTRAL_VALUES,
    SOURCE_ENCODING_DRIFT_BOUNDS,
    SOURCE_LEARNING_BOUNDS,
)

__all__ = ["prepare_work_unit", "review_results"]

WORK_UNIT = Path("work/pooled_model_runs")
TEMPLATE = Path("templates/fitting_evosaxde.ipynb")

DIAGNOSTICS = [
    {
        "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
        "figure_suffix": "category_recall",
        "kwargs": {
            "category_field": "condition",
            "category_values": [1, 2],
            "labels": ["Negative", "Neutral"],
        },
        "ylim": [0.2, 0.8],
        "color_cycle": ["red", "black"],
    },
    {
        "target": "jaxcmr.analyses.cat_lpp_by_recall.plot_cat_lpp_by_recall",
        "figure_suffix": "lpp_by_recall",
        "kwargs": {
            "category_field": "condition",
            "labels": [
                "Recalled Negative",
                "Unrecalled Negative",
                "Recalled Neutral",
                "Unrecalled Neutral",
            ],
            "category_value": [2, 1, 4, 3],
            "contrast_name": "Condition x Recall",
            "lpp_field": "EarlyLPP",
            "exclude_ci": True,
        },
        "ylim": [-0.6, 2.2],
    },
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assert_within(path: Path, parent: Path) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(f"Output escapes work unit: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_jaxcmr_root(explicit: str | Path | None) -> Path:
    candidate = explicit or os.environ.get("JAXCMR_ROOT")
    if candidate:
        root = Path(candidate).expanduser().resolve()
    else:
        import jaxcmr

        root = Path(jaxcmr.__file__).resolve().parents[1]
    template = root / TEMPLATE
    if not template.exists():
        raise FileNotFoundError(f"Missing fitting template: {template}")
    return root


def _fit_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": model["model_name"],
                "architecture": model["architecture"],
                "categorical_enhancement": model["categorical_enhancement"],
                "general_lpp": model["general_lpp"],
                "emotional_lpp": model["emotional_lpp"],
                "lpp_form": model["lpp_form"],
                "free_parameter_count": len(model["parameters"]["free"]),
                "free_parameters": ";".join(model["parameters"]["free"]),
                "free_parameter_bounds": json.dumps(
                    model["parameters"]["free"], sort_keys=True
                ),
                "fixed_parameters": json.dumps(
                    model["parameters"]["fixed"], sort_keys=True
                ),
                "model_factory": model["make_factory_path"],
            }
            for model in MODEL_COMPARISON_REGISTRY
        ]
    )


def _validate_registry() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(check: str, value: Any, expected: Any) -> None:
        passed = value == expected
        checks.append(
            {
                "check": check,
                "passed": passed,
                "value": value,
                "expected": expected,
            }
        )
        if not passed:
            raise AssertionError(f"{check}: expected {expected!r}, got {value!r}")

    names = tuple(model["model_name"] for model in MODEL_COMPARISON_REGISTRY)
    record(
        "model_names_match_issue_2",
        "|".join(names),
        "|".join(EXPECTED_MODEL_NAMES),
    )
    record("model_count", len(names), 16)
    record("unique_model_count", len(set(names)), 16)
    record(
        "post_update_learning",
        all(
            model["parameters"]["fixed"]["learn_after_context_update"]
            for model in MODEL_COMPARISON_REGISTRY
        ),
        True,
    )
    record("nesting_edge_count", len(NESTING_RELATIONSHIPS), 24)
    record("learning_strength_link", FIT_SETTINGS["learning_strength_link"], "log")
    record("lpp_slope_direction", FIT_SETTINGS["lpp_slope_direction"], "nonnegative")
    record("pooled", FIT_SETTINGS["pooled"], True)
    record(
        "termination_excluded_from_likelihood",
        "ExcludeTermination" in FIT_SETTINGS["loss_fn_path"],
        True,
    )
    record(
        "no_stop_policy",
        "NoStopTermination" in str(FIT_SETTINGS["component_paths"]),
        True,
    )

    expected_counts = [9, 10, 10, 11, 10, 11, 11, 12, 11, 12, 12, 13, 12, 13, 13, 14]
    actual_counts = [
        len(model["parameters"]["free"]) for model in MODEL_COMPARISON_REGISTRY
    ]
    record(
        "free_parameter_counts",
        ";".join(str(value) for value in actual_counts),
        ";".join(str(value) for value in expected_counts),
    )

    source_models = [
        model for model in MODEL_COMPARISON_REGISTRY if model["architecture"] == "source"
    ]
    temporal_models = [
        model
        for model in MODEL_COMPARISON_REGISTRY
        if model["architecture"] == "temporal"
    ]
    record(
        "source_learning_scale_free",
        all(
            model["parameters"]["free"].get("source_learning_rate")
            == SOURCE_LEARNING_BOUNDS
            for model in source_models
        ),
        True,
    )
    record(
        "source_encoding_drift_free",
        all(
            model["parameters"]["free"].get("source_encoding_drift_rate")
            == SOURCE_ENCODING_DRIFT_BOUNDS
            for model in source_models
        ),
        True,
    )
    record(
        "source_recall_drift_fixed_one",
        all(
            model["parameters"]["fixed"].get("source_recall_drift_rate") == 1.0
            for model in source_models
        ),
        True,
    )
    record(
        "source_parameters_absent_from_temporal_models",
        all(
            not {
                "source_learning_rate",
                "source_encoding_drift_rate",
                "source_recall_drift_rate",
            }
            & (model["parameters"]["fixed"].keys() | model["parameters"]["free"].keys())
            for model in temporal_models
        ),
        True,
    )
    record(
        "lpp_log_slope_bounds",
        all(
            bounds == LPP_SLOPE_BOUNDS
            for model in MODEL_COMPARISON_REGISTRY
            for parameter, bounds in model["parameters"]["free"].items()
            if parameter in {"lpp_main_scale", "lpp_inter_scale"}
        ),
        True,
    )

    lookup = {model["model_name"]: model for model in MODEL_COMPARISON_REGISTRY}
    exact_edges = True
    for edge in NESTING_RELATIONSHIPS:
        parent = lookup[edge["parent"]]["parameters"]
        child = lookup[edge["child"]]["parameters"]
        added = edge["added_parameter"]
        exact_edges &= set(child["free"]) == set(parent["free"]) | {added}
        exact_edges &= parent["fixed"].get(added) == PARAMETER_NEUTRAL_VALUES[added]
    record("nesting_edges_add_one_parameter", exact_edges, True)
    return checks


def _validate_data(project_root: Path) -> dict[str, Any]:
    data_path = project_root / FIT_SETTINGS["data_path"]
    data = load_data(data_path)
    mixed_data = slice_trials(
        data,
        mixed_trial_mask(data),
    )
    list_count = int(np.asarray(mixed_data["subject"]).shape[0])
    expected = FIT_SETTINGS["expected_list_count"]
    if list_count != expected:
        raise AssertionError(f"Expected {expected} lists, found {list_count}")
    early_lpp = np.asarray(mixed_data["EarlyLPP"], dtype=float)
    centered_lpp = early_lpp - early_lpp.mean(axis=1, keepdims=True)
    return {
        "list_count": list_count,
        "subject_count": int(np.unique(mixed_data["subject"]).size),
        "list_lengths": sorted(
            int(value) for value in np.unique(mixed_data["listLength"])
        ),
        "within_list_centered_early_lpp": {
            "minimum": float(centered_lpp.min()),
            "maximum": float(centered_lpp.max()),
            "standard_deviation": float(centered_lpp.std()),
        },
        "sha256": _sha256(data_path),
    }


def _validate_model_factories(project_root: Path) -> int:
    data = load_data(project_root / FIT_SETTINGS["data_path"])
    data = slice_trials(
        data,
        mixed_trial_mask(data),
    )
    components = {
        key: import_from_string(path)
        for key, path in FIT_SETTINGS["component_paths"].items()
    }
    validated = 0
    for config in MODEL_COMPARISON_REGISTRY:
        make_factory = import_from_string(config["make_factory_path"])
        factory = make_factory(**components)(data, None)
        parameters = dict(config["parameters"]["fixed"])
        parameters |= {
            name: (bounds[0] + bounds[1]) / 2
            for name, bounds in config["parameters"]["free"].items()
        }
        model = factory.create_trial_model(0, parameters)
        if not bool(model.learn_after_context_update):
            raise AssertionError(
                f"{config['model_name']} did not instantiate with post-update learning"
            )
        validated += 1
    return validated


def _template_parameters(model: dict[str, Any], unit_rel: str) -> dict[str, Any]:
    model_name = model["model_name"]
    best_of = FIT_SETTINGS["best_of"]
    return {
        "base_run_tag": FIT_SETTINGS["base_run_tag"],
        "experiment_count": FIT_SETTINGS["experiment_count"],
        "max_subjects": 0,
        "data_tag": FIT_SETTINGS["data_tag"],
        "data_path": FIT_SETTINGS["data_path"],
        "figure_dir": unit_rel,
        "figure_str": model_name,
        "embedding_path": "",
        "emotion_feature_path": "",
        "concat_features": False,
        "trial_query": FIT_SETTINGS["trial_query"],
        "target_directory": unit_rel,
        "fit_dir": unit_rel,
        "simulation_dir": unit_rel,
        "output_stem": f"{model_name}_best_of_{best_of}",
        "model_name": model_name,
        "make_factory_path": model["make_factory_path"],
        "component_paths": FIT_SETTINGS["component_paths"],
        "sim_alg_path": FIT_SETTINGS["sim_alg_path"],
        "loss_fn_path": FIT_SETTINGS["loss_fn_path"],
        "fit_alg_path": FIT_SETTINGS["fit_alg_path"],
        "parameters": model["parameters"],
        "subject_indices": [],
        "pooled": True,
        "filter_repeated_recalls": True,
        "redo_fits": False,
        "redo_sims": True,
        "redo_figures": True,
        "seed": FIT_SETTINGS["seed"],
        "relative_tolerance": FIT_SETTINGS["relative_tolerance"],
        "absolute_tolerance": FIT_SETTINGS["absolute_tolerance"],
        "popsize": FIT_SETTINGS["popsize"],
        "num_steps": FIT_SETTINGS["num_steps"],
        "cross_rate": FIT_SETTINGS["cross_rate"],
        "diff_w": list(FIT_SETTINGS["diff_w"]),
        "init": FIT_SETTINGS["init"],
        "best_of": best_of,
        "display_iterations": True,
        "comparison_analysis_configs": DIAGNOSTICS,
        "single_analysis_configs": [],
    }


def _make_notebook_query_cohort_aware(notebook: Any) -> None:
    """Freeze the query in the notebook and apply ``max_subjects`` after it.

    The shared jaxcmr template limits the loaded H5 before it evaluates
    ``trial_query``. In a combined pure/mixed file, that would select the
    lower-numbered pure-list participants first. Keep the full dataset loaded
    and restrict the already-selected cohort instead. Also replace the
    template's stale default query so the notebook has one visible cohort
    contract even when it is opened outside papermill.
    """

    setup_old = (
        "data = load_data(os.path.join(project_root, data_path), max_subjects)\n"
        "trial_mask = generate_trial_mask(data, trial_query)\n"
    )
    setup_new = (
        "data = load_data(os.path.join(project_root, data_path), 0)\n"
        "trial_mask = generate_trial_mask(data, trial_query)\n"
        "if max_subjects:\n"
        "    cohort_subjects = jnp.unique(\n"
        '        jnp.asarray(data["subject"]).reshape(-1)[trial_mask]\n'
        "    )[:max_subjects]\n"
        "    trial_mask = trial_mask & jnp.isin(\n"
        '        jnp.asarray(data["subject"]).reshape(-1), cohort_subjects\n'
        "    )\n"
    )
    simulation_old = 'unique_subjects = jnp.unique(jnp.array(data["subject"]))'
    simulation_new = (
        'unique_subjects = jnp.unique(jnp.asarray(data["subject"])'
        ".reshape(-1)[trial_mask])"
    )
    parameter_query_replacements = 0
    setup_replacements = 0
    simulation_replacements = 0
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = str(cell.get("source", ""))
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            lines = source.splitlines(keepends=True)
            query_lines = [
                index
                for index, line in enumerate(lines)
                if line.lstrip().startswith("trial_query = ")
            ]
            if len(query_lines) != 1:
                raise AssertionError(
                    "Expected one trial_query assignment in the parameters cell; "
                    f"found {len(query_lines)}"
                )
            index = query_lines[0]
            newline = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"trial_query = {FIT_SETTINGS['trial_query']!r}{newline}"
            source = "".join(lines)
            parameter_query_replacements += 1
        if setup_old in source:
            source = source.replace(setup_old, setup_new, 1)
            setup_replacements += 1
        if simulation_old in source:
            source = source.replace(simulation_old, simulation_new, 1)
            simulation_replacements += 1
        cell["source"] = source

    if (
        parameter_query_replacements != 1
        or setup_replacements != 1
        or simulation_replacements != 1
    ):
        raise AssertionError(
            "Could not make generated notebook cohort-aware: "
            f"query={parameter_query_replacements}, setup={setup_replacements}, "
            f"simulation={simulation_replacements}"
        )


def _expected_products(model_name: str) -> dict[str, str]:
    best_of = FIT_SETTINGS["best_of"]
    return {
        "notebook": f"fit_{model_name}.ipynb",
        "fit": f"{model_name}_best_of_{best_of}.json",
        "simulation": f"{model_name}_best_of_{best_of}.h5",
        "category_recall": f"{model_name}_category_recall.png",
        "lpp_by_recall": f"{model_name}_lpp_by_recall.png",
    }


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare_work_unit(jaxcmr_root: str | Path | None = None) -> dict[str, Any]:
    """Generate the 16 cluster-ready notebooks without executing fits."""

    import nbformat
    import papermill as pm

    project_root = _project_root()
    unit_dir = project_root / WORK_UNIT
    unit_dir.mkdir(parents=True, exist_ok=True)
    _assert_within(unit_dir, project_root / "work")
    unit_rel = _relative(unit_dir, project_root)

    resolved_jaxcmr_root = _resolve_jaxcmr_root(jaxcmr_root)
    template = resolved_jaxcmr_root / TEMPLATE
    template_notebook = nbformat.read(template, as_version=4)
    template_source = "\n".join(
        "".join(cell.get("source", "")) for cell in template_notebook.cells
    )
    for parameter in ("fit_dir", "simulation_dir", "figure_dir", "output_stem"):
        if parameter not in template_source:
            raise AssertionError(
                f"jaxcmr template does not support flat output parameter {parameter!r}"
            )
    if '"figures": figure_dir' not in template_source:
        raise AssertionError("jaxcmr template does not route figures to figure_dir")
    if "exist_ok=True" not in template_source:
        raise AssertionError("jaxcmr template directory creation is not concurrency-safe")
    if "figures/fitting" in template_source:
        raise AssertionError("jaxcmr template retains the legacy nested figure directory")

    checks = _validate_registry()
    data_summary = _validate_data(project_root)
    checks.append(
        {
            "check": "list_count",
            "passed": True,
            "value": data_summary["list_count"],
            "expected": FIT_SETTINGS["expected_list_count"],
        }
    )
    lpp_summary = data_summary["within_list_centered_early_lpp"]
    for name, observed, expected in (
        ("centered_lpp_minimum", lpp_summary["minimum"], LPP_CENTERED_MIN),
        ("centered_lpp_maximum", lpp_summary["maximum"], LPP_CENTERED_MAX),
    ):
        checks.append(
            {
                "check": name,
                "passed": bool(np.isclose(observed, expected)),
                "value": observed,
                "expected": expected,
            }
        )
        if not checks[-1]["passed"]:
            raise AssertionError(f"{name}: expected {expected}, found {observed}")
    checks.append(
        {
            "check": "flat_output_template_parameters",
            "passed": True,
            "value": "fit_dir;simulation_dir;figure_dir;output_stem;exist_ok",
            "expected": "fit_dir;simulation_dir;figure_dir;output_stem;exist_ok",
        }
    )
    factory_count = _validate_model_factories(project_root)
    checks.append(
        {
            "check": "model_factories_instantiated",
            "passed": factory_count == 16,
            "value": factory_count,
            "expected": 16,
        }
    )

    fit_grid = _fit_grid()
    fit_grid.to_csv(unit_dir / "fit_grid.csv", index=False)
    _write_csv(checks, unit_dir / "unit_checks.csv")

    plot_rows = []
    for model in MODEL_COMPARISON_REGISTRY:
        name = model["model_name"]
        products = _expected_products(name)
        output_path = unit_dir / products["notebook"]
        _assert_within(output_path, unit_dir)
        parameters = _template_parameters(model, unit_rel)

        pm.execute_notebook(
            str(template),
            str(output_path),
            parameters=parameters,
            progress_bar=False,
            prepare_only=True,
        )
        notebook = nbformat.read(output_path, as_version=4)
        _make_notebook_query_cohort_aware(notebook)
        papermill_metadata = notebook.metadata.get("papermill", {})
        papermill_metadata["input_path"] = TEMPLATE.as_posix()
        papermill_metadata["output_path"] = products["notebook"]
        notebook.metadata["papermill"] = papermill_metadata
        nbformat.write(notebook, output_path)
        source = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
        if "/Users/" in source:
            raise AssertionError(f"Local absolute path in {output_path.name}")
        injected = notebook.metadata.get("papermill", {}).get("parameters", {})
        for parameter in ("fit_dir", "simulation_dir", "figure_dir"):
            if injected.get(parameter) != unit_rel:
                raise AssertionError(
                    f"{output_path.name} has invalid {parameter}: "
                    f"{injected.get(parameter)!r}"
                )
        expected_injected = {
            "trial_query": FIT_SETTINGS["trial_query"],
            "subject_indices": [],
            "pooled": True,
        }
        for parameter, expected_value in expected_injected.items():
            if injected.get(parameter) != expected_value:
                raise AssertionError(
                    f"{output_path.name} has invalid {parameter}: "
                    f"{injected.get(parameter)!r}"
                )
        if "/Users/" in output_path.read_text():
            raise AssertionError(f"Local absolute metadata path in {output_path.name}")

        for product_type, filename in products.items():
            if product_type not in {"category_recall", "lpp_by_recall"}:
                continue
            path = unit_dir / filename
            _assert_within(path, unit_dir)
            plot_rows.append(
                {
                    "model_name": name,
                    "diagnostic": product_type,
                    "path": filename,
                    "exists": path.exists(),
                }
            )

    _write_csv(plot_rows, unit_dir / "plot_manifest.csv")

    return {
        "work_unit": WORK_UNIT.as_posix(),
        "model_count": len(MODEL_COMPARISON_REGISTRY),
        "generated_notebooks": len(MODEL_COMPARISON_REGISTRY),
        "fit_grid": "fit_grid.csv",
        "unit_checks": "unit_checks.csv",
        "plot_inventory": "plot_manifest.csv",
    }


def _first_scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size else np.nan


def _learning_strength_diagnostics(
    model: dict[str, Any], parameters: dict[str, Any], data: dict[str, Any]
) -> dict[str, float]:
    """Summarize the fitted pathway carrying categorical and LPP modulation."""

    early_lpp = np.asarray(data["EarlyLPP"], dtype=float)
    centered_lpp = early_lpp - early_lpp.mean(axis=1, keepdims=True)
    is_emotional = (2 - np.asarray(data["condition"], dtype=float)).astype(bool)
    study_index = np.arange(centered_lpp.shape[1], dtype=float)[None, :]
    primacy = parameters["primacy_scale"] * np.exp(
        -parameters["primacy_decay"] * study_index
    ) + 1.0
    pathway_scale = (
        parameters["source_learning_rate"]
        if model["architecture"] == "source"
        else 1.0
    )
    categorical_multiplier = 1.0 + (
        parameters.get("emotion_scale", 1.0) - 1.0
    ) * is_emotional
    log_lpp_multiplier = (
        parameters.get("lpp_main_scale", 0.0) * centered_lpp
        + parameters.get("lpp_inter_scale", 0.0)
        * is_emotional
        * centered_lpp
    )
    lpp_multiplier = np.exp(log_lpp_multiplier)
    learning_strength = (
        pathway_scale * primacy * categorical_multiplier * lpp_multiplier
    )
    if not np.isfinite(learning_strength).all():
        raise ValueError(
            f"{model['model_name']} produced non-finite learning-strength diagnostics"
        )
    return {
        "minimum_learning_strength": float(learning_strength.min()),
        "maximum_learning_strength": float(learning_strength.max()),
        "minimum_lpp_multiplier": float(lpp_multiplier.min()),
        "maximum_lpp_multiplier": float(lpp_multiplier.max()),
    }


def review_results() -> pd.DataFrame:
    """Aggregate completed fit files and require every expected product."""

    project_root = _project_root()
    unit_dir = project_root / WORK_UNIT
    comparison_rows = []
    parameter_rows = []
    results_by_model = {}
    data = load_data(project_root / FIT_SETTINGS["data_path"])
    data = slice_trials(
        data,
        mixed_trial_mask(data),
    )

    for model in MODEL_COMPARISON_REGISTRY:
        name = model["model_name"]
        products = _expected_products(name)
        missing = [
            filename
            for product, filename in products.items()
            if product != "notebook" and not (unit_dir / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(f"{name} is incomplete: {missing}")

        with (unit_dir / products["fit"]).open() as handle:
            result = json.load(handle)
        nll = float(_first_scalar(result["fitness"]))
        if not np.isfinite(nll):
            raise ValueError(f"{name} has non-finite NLL: {nll}")
        free = result.get("free", model["parameters"]["free"])
        parameter_count = len(free)
        fitted_parameters = dict(model["parameters"]["fixed"])
        fitted_parameters |= {
            parameter: _first_scalar(value)
            for parameter, value in result["fits"].items()
            if parameter != "subject"
        }
        learning_diagnostics = _learning_strength_diagnostics(
            model, fitted_parameters, data
        )
        results_by_model[name] = {"NLL": nll, "result": result}
        comparison_rows.append(
            {
                "model_name": name,
                "architecture": model["architecture"],
                "parameter_count": parameter_count,
                "NLL": nll,
                "AIC": 2 * nll + 2 * parameter_count,
                **learning_diagnostics,
            }
        )
        for parameter, value in result["fits"].items():
            if parameter == "subject":
                continue
            estimate = _first_scalar(value)
            if not np.isfinite(estimate):
                raise ValueError(f"{name} has non-finite {parameter}: {estimate}")
            bounds = model["parameters"]["free"].get(parameter)
            if bounds and not bounds[0] <= estimate <= bounds[1]:
                raise ValueError(
                    f"{name} has out-of-bounds {parameter}: {estimate} not in {bounds}"
                )
            parameter_rows.append(
                {
                    "model_name": name,
                    "parameter": parameter,
                    "manuscript_parameter": PARAMETER_MANUSCRIPT_NAMES.get(
                        parameter, parameter
                    ),
                    "estimate": estimate,
                }
            )

    nesting_violations = []
    for edge in NESTING_RELATIONSHIPS:
        parent_nll = results_by_model[edge["parent"]]["NLL"]
        child_nll = results_by_model[edge["child"]]["NLL"]
        if child_nll > parent_nll + 1e-6:
            nesting_violations.append(
                {
                    "parent": edge["parent"],
                    "child": edge["child"],
                    "parent_NLL": parent_nll,
                    "child_NLL": child_nll,
                }
            )
    if nesting_violations:
        raise ValueError(
            "Optimizer-related nesting violations must be resolved before review: "
            f"{nesting_violations}"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("AIC")
    comparison["delta_AIC"] = comparison["AIC"] - comparison["AIC"].min()
    comparison.to_csv(unit_dir / "model_comparison.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(unit_dir / "parameter_summary.csv", index=False)

    product_manifest = pd.read_csv(unit_dir / "plot_manifest.csv")
    product_manifest["exists"] = product_manifest["path"].map(
        lambda path: (unit_dir / path).exists()
    )
    if not product_manifest["exists"].all():
        raise AssertionError("At least one expected product is missing")
    product_manifest.to_csv(unit_dir / "plot_manifest.csv", index=False)

    return comparison


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxcmr-root")
    args = parser.parse_args()
    summary = prepare_work_unit(args.jaxcmr_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
