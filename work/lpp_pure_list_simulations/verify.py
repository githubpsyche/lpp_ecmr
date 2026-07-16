"""Verify the completed pure-list simulation work package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "work" / "lpp_model_comparison"
EXPECTED_MODELS = 16
EXPECTED_TRIALS_PER_TYPE = 171 * 200
EXPECTED_RECALLS_PER_TYPE = 1537 * 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    notebooks = sorted(WORK_DIR.glob("fit_*.ipynb"))
    fits = sorted(WORK_DIR.glob("*_best_of_3.json"))
    simulations = sorted(WORK_DIR.glob("*_best_of_3.h5"))
    figures = sorted(WORK_DIR.glob("*.png"))

    if len(notebooks) != EXPECTED_MODELS:
        raise AssertionError(f"Expected {EXPECTED_MODELS} notebooks, found {len(notebooks)}")
    if len(fits) != EXPECTED_MODELS:
        raise AssertionError(f"Expected {EXPECTED_MODELS} fits, found {len(fits)}")
    if len(simulations) != EXPECTED_MODELS:
        raise AssertionError(
            f"Expected {EXPECTED_MODELS} simulations, found {len(simulations)}"
        )
    if len(figures) != EXPECTED_MODELS * 2:
        raise AssertionError(f"Expected {EXPECTED_MODELS * 2} figures, found {len(figures)}")

    notebook_checks: dict[str, dict[str, object]] = {}
    for notebook in notebooks:
        document = json.loads(notebook.read_text())
        papermill = document.get("metadata", {}).get("papermill", {})
        if papermill.get("exception"):
            raise AssertionError(f"Notebook execution failed: {notebook.name}")
        for cell in document["cells"]:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    raise AssertionError(f"Error output in {notebook.name}")
        source = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])
        if "redo_fits = False" not in source:
            raise AssertionError(f"{notebook.name} is not configured to reuse its fit")
        if "redo_sims = True" not in source:
            raise AssertionError(f"{notebook.name} is not configured to reproduce its simulation")
        if '"datasets": [sim]' not in source:
            raise AssertionError(f"{notebook.name} does not draw pure-simulation figures")
        notebook_checks[notebook.name] = {
            "execution_seconds": papermill.get("duration"),
            "exception": False,
        }

    fit_checks: dict[str, str] = {}
    for fit in fits:
        source = SOURCE_DIR / fit.name
        if not source.exists():
            raise FileNotFoundError(source)
        copied_hash = _sha256(fit)
        if copied_hash != _sha256(source):
            raise AssertionError(f"Copied fit differs from source: {fit.name}")
        fit_checks[fit.name] = copied_hash

    simulation_checks: dict[str, dict[str, int | bool]] = {}
    for simulation in simulations:
        with h5py.File(simulation) as handle:
            data_group = handle["data"]
            list_type = np.asarray(data_group["list_type"]).reshape(-1)
            # save_dict_to_hdf5 stores trial-major arrays transposed on disk.
            condition = np.asarray(data_group["condition"]).T
            recalls = np.asarray(data_group["recalls"]).T
        counts = {value: int(np.count_nonzero(list_type == value)) for value in (1, 2)}
        recall_totals = {
            value: int(np.count_nonzero(recalls[list_type == value])) for value in (1, 2)
        }
        if counts != {1: EXPECTED_TRIALS_PER_TYPE, 2: EXPECTED_TRIALS_PER_TYPE}:
            raise AssertionError(f"Unbalanced list types in {simulation.name}: {counts}")
        if recall_totals != {1: EXPECTED_RECALLS_PER_TYPE, 2: EXPECTED_RECALLS_PER_TYPE}:
            raise AssertionError(
                f"Unbalanced forced recall totals in {simulation.name}: {recall_totals}"
            )
        pure_negative = bool(np.all(condition[list_type == 1] == 1))
        pure_neutral = bool(np.all(condition[list_type == 2] == 2))
        if not pure_negative or not pure_neutral:
            raise AssertionError(f"Non-pure list found in {simulation.name}")
        simulation_checks[simulation.name] = {
            "pure_negative_trials": counts[1],
            "pure_neutral_trials": counts[2],
            "pure_negative_recall_events": recall_totals[1],
            "pure_neutral_recall_events": recall_totals[2],
            "all_lists_pure": True,
        }

    figure_checks: dict[str, list[int]] = {}
    for figure in figures:
        with Image.open(figure) as image:
            image.verify()
        with Image.open(figure) as image:
            figure_checks[figure.name] = [int(image.width), int(image.height)]

    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "notebook_count": len(notebooks),
        "fit_count": len(fits),
        "simulation_count": len(simulations),
        "figure_count": len(figures),
        "notebooks": notebook_checks,
        "fits": fit_checks,
        "simulations": simulation_checks,
        "figures": figure_checks,
    }
    (WORK_DIR / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Verified {len(notebooks)} notebooks, {len(fits)} unchanged fits, "
        f"{len(simulations)} pure-list simulations, and {len(figures)} figures."
    )


if __name__ == "__main__":
    main()
