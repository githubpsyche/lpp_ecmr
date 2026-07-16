"""Execute all prepared pure-list simulation notebooks sequentially."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import papermill as pm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent


def _set_redo_sims(notebook: Path, value: bool) -> None:
    document = json.loads(notebook.read_text())
    replacement = f"redo_sims = {value}"
    changed = False
    for cell in document["cells"]:
        if cell.get("cell_type") != "code" or "injected-parameters" not in cell.get(
            "metadata", {}
        ).get("tags", []):
            continue
        source = "".join(cell.get("source", []))
        for current in ("redo_sims = True", "redo_sims = False"):
            if current in source:
                source = source.replace(current, replacement, 1)
                cell["source"] = source.splitlines(keepends=True)
                changed = True
                break
    parameters = document.get("metadata", {}).get("papermill", {}).get("parameters", {})
    if "redo_sims" in parameters:
        parameters["redo_sims"] = value
    if not changed:
        raise ValueError(f"Could not set redo_sims in {notebook.name}")
    notebook.write_text(json.dumps(document, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-simulations",
        action="store_true",
        help="Load existing H5 files and rerender figures without simulating again.",
    )
    args = parser.parse_args()

    notebooks = sorted(WORK_DIR.glob("fit_*.ipynb"))
    if len(notebooks) != 16:
        raise ValueError(f"Expected 16 fit notebooks, found {len(notebooks)}.")

    for index, notebook in enumerate(notebooks, start=1):
        temporary = notebook.with_suffix(".executing.ipynb")
        started = time.monotonic()
        print(f"[{index:02d}/{len(notebooks):02d}] Executing {notebook.name}", flush=True)
        try:
            if args.reuse_simulations:
                _set_redo_sims(notebook, False)
            pm.execute_notebook(
                notebook,
                temporary,
                kernel_name="workspace",
                cwd=PROJECT_ROOT,
                log_output=True,
                progress_bar=False,
            )
            os.replace(temporary, notebook)
            if args.reuse_simulations:
                # Leave the delivered notebook configured to reproduce the full
                # pure-list simulation, while retaining the executed outputs.
                _set_redo_sims(notebook, True)
        finally:
            if temporary.exists():
                temporary.unlink()
            if args.reuse_simulations and notebook.exists():
                _set_redo_sims(notebook, True)
        elapsed = time.monotonic() - started
        print(f"[{index:02d}/{len(notebooks):02d}] Finished in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
