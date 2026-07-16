"""Prepare pure-list simulation notebooks from the pooled fitting package."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from jaxcmr.helpers import load_data
from scipy.optimize import Bounds, LinearConstraint, milp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
TARGET_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"
PACKAGE_PATH = "work/lpp_pure_list_simulations"


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _make_design() -> list[dict[str, int | str]]:
    """Make an exactly balanced, participant-stratified pure-list design."""
    data = load_data(DATA_PATH)
    subjects = np.asarray(data["subject"]).reshape(-1).astype(int)
    lists = np.asarray(data["list"]).reshape(-1).astype(int)
    recall_counts = np.count_nonzero(np.asarray(data["recalls"]), axis=1)
    unique_subjects = np.unique(subjects)
    trial_count = subjects.size
    subject_count = unique_subjects.size

    if trial_count % 2:
        raise ValueError("The list count must be even for an equal pure-list split.")
    if int(recall_counts.sum()) % 2:
        raise ValueError("Recall events cannot be balanced exactly between list types.")

    # Binary variables choose pure-negative trials. Continuous deviation
    # variables minimize within-participant recall-count imbalance.
    variable_count = trial_count + subject_count
    objective = np.r_[np.zeros(trial_count), np.ones(subject_count)]
    integrality = np.r_[np.ones(trial_count), np.zeros(subject_count)]
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.r_[np.ones(trial_count), np.full(subject_count, np.inf)]

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    row = np.zeros(variable_count)
    row[:trial_count] = 1
    rows.append(row)
    lower.append(trial_count / 2)
    upper.append(trial_count / 2)

    row = np.zeros(variable_count)
    row[:trial_count] = recall_counts
    rows.append(row)
    lower.append(recall_counts.sum() / 2)
    upper.append(recall_counts.sum() / 2)

    for subject_index, subject in enumerate(unique_subjects):
        subject_mask = subjects == subject
        subject_trial_count = int(subject_mask.sum())
        subject_recall_count = int(recall_counts[subject_mask].sum())

        row = np.zeros(variable_count)
        row[:trial_count] = subject_mask
        rows.append(row)
        lower.append(subject_trial_count // 2)
        upper.append((subject_trial_count + 1) // 2)

        # d_s >= abs(2 * negative_recall_count_s - total_recall_count_s)
        row = np.zeros(variable_count)
        row[:trial_count] = 2 * recall_counts * subject_mask
        row[trial_count + subject_index] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(subject_recall_count)

        row = np.zeros(variable_count)
        row[:trial_count] = -2 * recall_counts * subject_mask
        row[trial_count + subject_index] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(-subject_recall_count)

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Could not construct pure-list design: {result.message}")

    pure_negative = np.rint(result.x[:trial_count]).astype(bool)
    if pure_negative.sum() != trial_count // 2:
        raise AssertionError("Pure-list trial counts are not balanced.")
    if recall_counts[pure_negative].sum() != recall_counts[~pure_negative].sum():
        raise AssertionError("Recall-count masks are not balanced.")

    rows_out: list[dict[str, int | str]] = []
    for trial_index in range(trial_count):
        is_negative = bool(pure_negative[trial_index])
        rows_out.append(
            {
                "trial_index": trial_index,
                "subject": int(subjects[trial_index]),
                "list": int(lists[trial_index]),
                "recall_count": int(recall_counts[trial_index]),
                "list_type": "pure_negative" if is_negative else "pure_neutral",
                "condition": 1 if is_negative else 2,
            }
        )
    return rows_out


def _write_design(rows: list[dict[str, int | str]]) -> None:
    path = TARGET_DIR / "pure_list_design.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adapt_notebook(source_path: Path, target_path: Path) -> None:
    notebook = json.loads(source_path.read_text())
    cells = notebook["cells"]

    _set_source(
        cells[0],
        "Use this notebook to reuse one pooled fit, simulate equal numbers of "
        "pure-negative and pure-neutral lists, and generate benchmark diagnostics.\n",
    )

    # The imports are shared by every rendered fitting notebook.
    import_source = _source(cells[1])
    if "import csv\n" not in import_source:
        import_source = import_source.replace("import inspect\n", "import csv\nimport inspect\n")
    _set_source(cells[1], import_source)

    parameter_source = _source(cells[4]).replace(
        "work/lpp_model_comparison", PACKAGE_PATH
    )
    parameter_source = parameter_source.replace(
        '"labels": ["Negative", "Neutral"]',
        '"labels": ["Pure negative", "Pure neutral"]',
    )
    parameter_source = parameter_source.replace(
        '"color_cycle": ["red", "black"]',
        '"color_cycle": ["#C44E52", "#4C72B0"]',
    )
    parameter_source = parameter_source.replace(
        '"labels": ["Recalled Negative", "Unrecalled Negative", '
        '"Recalled Neutral", "Unrecalled Neutral"]',
        '"labels": ["Recalled negative", "Unrecalled negative", '
        '"Recalled neutral", "Unrecalled neutral"]',
    )
    _set_source(cells[4], parameter_source)

    setup_source = _source(cells[5])
    marker = "trial_mask = generate_trial_mask(data, trial_query)\n"
    if marker not in setup_source:
        raise ValueError(f"Could not locate data setup in {source_path.name}")
    pure_list_setup = '''

# Construct the shared pure-list simulation inputs while retaining the original
# mixed-list dataset for loading the pooled fit and drawing empirical curves.
pure_list_design_path = project_root / "work/lpp_pure_list_simulations/pure_list_design.csv"
with pure_list_design_path.open(newline="") as handle:
    pure_list_design = list(csv.DictReader(handle))

trial_count = np.asarray(data["subject"]).shape[0]
design_trial_indices = np.array(
    [int(row["trial_index"]) for row in pure_list_design], dtype=np.int32
)
design_subjects = np.array(
    [int(row["subject"]) for row in pure_list_design], dtype=np.int32
)
design_lists = np.array(
    [int(row["list"]) for row in pure_list_design], dtype=np.int32
)
pure_conditions = np.array(
    [int(row["condition"]) for row in pure_list_design], dtype=np.int32
)

if not np.array_equal(design_trial_indices, np.arange(trial_count)):
    raise ValueError("pure_list_design.csv does not match the data trial order.")
if not np.array_equal(design_subjects, np.asarray(data["subject"]).reshape(-1)):
    raise ValueError("pure_list_design.csv does not match the data subjects.")
if not np.array_equal(design_lists, np.asarray(data["list"]).reshape(-1)):
    raise ValueError("pure_list_design.csv does not match the data list identifiers.")

simulation_fields = (
    "EarlyLPP",
    "list",
    "listLength",
    "pres_itemids",
    "pres_itemnos",
    "recalls",
    "subject",
)
simulation_data = {
    key: np.array(data[key], copy=True) for key in simulation_fields
}
simulation_data["condition"] = np.repeat(
    pure_conditions[:, np.newaxis],
    np.asarray(data["condition"]).shape[1],
    axis=1,
).astype(np.int32)
simulation_data["list_type"] = pure_conditions[:, np.newaxis]

if np.count_nonzero(pure_conditions == 1) != trial_count // 2:
    raise AssertionError("Expected exactly half pure-negative trials.")
if np.count_nonzero(pure_conditions == 2) != trial_count // 2:
    raise AssertionError("Expected exactly half pure-neutral trials.")
if not np.all(simulation_data["condition"][pure_conditions == 1] == 1):
    raise AssertionError("A pure-negative trial contains a neutral item.")
if not np.all(simulation_data["condition"][pure_conditions == 2] == 2):
    raise AssertionError("A pure-neutral trial contains a negative item.")

print(
    f"Pure-list simulation design: {np.count_nonzero(pure_conditions == 1)} "
    f"negative and {np.count_nonzero(pure_conditions == 2)} neutral lists"
)
'''
    setup_source = setup_source.replace(marker, marker + pure_list_setup, 1)
    _set_source(cells[5], setup_source)

    simulation_source = _source(cells[9])
    simulation_source = simulation_source.replace(
        'jnp.unique(jnp.array(data["subject"]))',
        'jnp.unique(jnp.array(simulation_data["subject"]))',
    )
    old_call = """            model_factory_cls,
            data,
            modeling_features,"""
    new_call = """            model_factory_cls,
            simulation_data,
            modeling_features,"""
    if old_call not in simulation_source:
        raise ValueError(f"Could not locate simulation call in {source_path.name}")
    simulation_source = simulation_source.replace(old_call, new_call, 1)
    _set_source(cells[9], simulation_source)

    _set_source(
        cells[10],
        "Figures summarize the pure-list simulations.\n",
    )

    figure_source = _source(cells[12])
    figure_source = figure_source.replace(
        '"datasets": [sim, data],\n            "trial_masks": '
        '[np.array(sim_trial_mask), np.array(trial_mask)],',
        '"datasets": [sim],\n            "trial_masks": [np.array(sim_trial_mask)],',
    )
    _set_source(cells[12], figure_source)

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    parameters = notebook.get("metadata", {}).get("papermill", {}).get("parameters", {})
    for key in ("figure_dir", "target_directory", "fit_dir", "simulation_dir"):
        if key in parameters:
            parameters[key] = PACKAGE_PATH
    parameters["redo_fits"] = False
    parameters["redo_sims"] = True
    parameters["redo_figures"] = True
    for config in parameters.get("comparison_analysis_configs", []):
        if config.get("figure_suffix") == "category_recall":
            config["kwargs"]["labels"] = [
                "Pure negative",
                "Pure neutral",
            ]
            config["color_cycle"] = ["#C44E52", "#4C72B0"]
        elif config.get("figure_suffix") == "lpp_by_recall":
            config["kwargs"]["labels"] = [
                "Recalled negative",
                "Unrecalled negative",
                "Recalled neutral",
                "Unrecalled neutral",
            ]

    target_path.write_text(json.dumps(notebook, indent=1) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    design = _make_design()
    _write_design(design)

    source_notebooks = sorted(SOURCE_DIR.glob("fit_*.ipynb"))
    if len(source_notebooks) != 16:
        raise ValueError(f"Expected 16 fit notebooks, found {len(source_notebooks)}.")

    manifest: dict[str, object] = {
        "source_directory": str(SOURCE_DIR.relative_to(PROJECT_ROOT)),
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "notebook_count": len(source_notebooks),
        "experiment_count": 200,
        "pure_negative_trials": sum(row["condition"] == 1 for row in design),
        "pure_neutral_trials": sum(row["condition"] == 2 for row in design),
        "fits": {},
    }

    for source_notebook in source_notebooks:
        model = source_notebook.stem.removeprefix("fit_")
        source_fit = SOURCE_DIR / f"{model}_best_of_3.json"
        if not source_fit.exists():
            raise FileNotFoundError(source_fit)

        target_notebook = TARGET_DIR / source_notebook.name
        target_fit = TARGET_DIR / source_fit.name
        _adapt_notebook(source_notebook, target_notebook)
        shutil.copy2(source_fit, target_fit)
        manifest["fits"][model] = {
            "filename": target_fit.name,
            "sha256": _sha256(target_fit),
        }

    (TARGET_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Prepared {len(source_notebooks)} notebooks in {TARGET_DIR}")


if __name__ == "__main__":
    main()
