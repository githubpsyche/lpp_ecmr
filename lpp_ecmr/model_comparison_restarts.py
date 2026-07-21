"""Run pooled model-comparison optimizer restarts as independent cluster tasks.

The scientific comparison still uses ``best_of=3``.  This module changes only
the scheduling: each of the 48 model/restart combinations is fit separately
with the exact key that the ordinary sequential ``EvosaxDE.fit`` call would
have used.  A later reduction selects one complete minimum-NLL restart per
model; parameter vectors are never averaged.

The task grid is derived directly from the executable model registry.  No
persisted task manifest is required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

import jaxcmr
from jaxcmr.fitting import EvosaxDE
from jaxcmr.helpers import import_from_string, load_data

from .data_contract import (
    MIXED_EXPECTED_LISTS,
    MIXED_EXPECTED_SUBJECTS,
    MIXED_TRIAL_QUERY,
    mixed_trial_mask,
    validate_combined_dataset,
)
from .model_comparison_registry import (
    FIT_SETTINGS,
    MODEL_COMPARISON_REGISTRY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "work" / "pooled_model_runs"
DEFAULT_DATA_PATH = PROJECT_ROOT / FIT_SETTINGS["data_path"]

SCHEMA_VERSION = 1
FIT_CALL_INDEX = 0
RESTART_COUNT = int(FIT_SETTINGS["best_of"])
TASK_COUNT = len(MODEL_COMPARISON_REGISTRY) * RESTART_COUNT

if RESTART_COUNT != 3:
    raise AssertionError(f"Expected best_of=3; found {RESTART_COUNT}.")
if len(MODEL_COMPARISON_REGISTRY) != 16:
    raise AssertionError(
        f"Expected 16 model-comparison cells; found {len(MODEL_COMPARISON_REGISTRY)}."
    )


class RestartError(RuntimeError):
    """Raised when a restart task or reduction violates its contract."""


@dataclass(frozen=True)
class RestartTask:
    """One independently schedulable model/restart optimization."""

    task_index: int
    model_index: int
    restart_index: int
    model_name: str
    restart_key: tuple[int, int]

    @property
    def filename(self) -> str:
        """Return the unique flat output filename for this restart."""

        return f"{self.model_name}_restart_{self.restart_index}.json"


def _json_copy(value: Any) -> Any:
    """Return a JSON-shaped copy, normalizing tuples and NumPy scalars."""

    return json.loads(json.dumps(value))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restart_keys() -> tuple[tuple[int, int], ...]:
    """Return keys from the first ordinary pooled ``best_of`` fit call."""

    base_key = jax.random.PRNGKey(int(FIT_SETTINGS["seed"]))
    fit_key = jax.random.fold_in(base_key, FIT_CALL_INDEX)
    keys = np.asarray(jax.random.split(fit_key, RESTART_COUNT), dtype=np.uint32)
    return tuple(tuple(int(word) for word in row) for row in keys)


def restart_tasks() -> tuple[RestartTask, ...]:
    """Return the complete deterministic 16-model by 3-restart task grid."""

    keys = _restart_keys()
    tasks = []
    for task_index in range(TASK_COUNT):
        model_index, restart_index = divmod(task_index, RESTART_COUNT)
        model = MODEL_COMPARISON_REGISTRY[model_index]
        tasks.append(
            RestartTask(
                task_index=task_index,
                model_index=model_index,
                restart_index=restart_index,
                model_name=str(model["model_name"]),
                restart_key=keys[restart_index],
            )
        )
    return tuple(tasks)


def task_for_index(task_index: int) -> RestartTask:
    """Resolve one Slurm array index to its immutable task specification."""

    if not 0 <= task_index < TASK_COUNT:
        raise RestartError(
            f"task_index must be in 0..{TASK_COUNT - 1}; found {task_index}."
        )
    return restart_tasks()[task_index]


def _model_for_task(task: RestartTask) -> dict[str, Any]:
    """Return the registry entry fixed by ``task``."""

    model = MODEL_COMPARISON_REGISTRY[task.model_index]
    if model["model_name"] != task.model_name:
        raise RestartError(
            f"Task {task.task_index} no longer matches registry model "
            f"{task.model_name!r}."
        )
    return model


def _source_hashes() -> dict[str, str]:
    """Hash executable scientific sources used by every restart."""

    jaxcmr_root = Path(jaxcmr.__file__).resolve().parents[1]
    relative_paths = {
        "lpp_ecmr/model_comparison_registry.py": (
            PROJECT_ROOT / "lpp_ecmr/model_comparison_registry.py"
        ),
        "lpp_ecmr/models/learning_strength.py": (
            PROJECT_ROOT / "lpp_ecmr/models/learning_strength.py"
        ),
        "lpp_ecmr/models/single_eeg_ecmr.py": (
            PROJECT_ROOT / "lpp_ecmr/models/single_eeg_ecmr.py"
        ),
        "lpp_ecmr/models/full_eeg_ecmr.py": (
            PROJECT_ROOT / "lpp_ecmr/models/full_eeg_ecmr.py"
        ),
        "jaxcmr/components/context.py": jaxcmr_root / "jaxcmr/components/context.py",
        "jaxcmr/fitting/evosax.py": jaxcmr_root / "jaxcmr/fitting/evosax.py",
        "jaxcmr/loss/set_permutation_likelihood.py": (
            jaxcmr_root / "jaxcmr/loss/set_permutation_likelihood.py"
        ),
    }
    missing = [name for name, path in relative_paths.items() if not path.is_file()]
    if missing:
        raise RestartError(f"Missing executable source files: {missing}.")
    return {name: _sha256(path) for name, path in relative_paths.items()}


def _fitter_hyperparameters(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical pooled EvosaxDE configuration."""

    return {
        "num_steps": FIT_SETTINGS["num_steps"],
        "pop_size": FIT_SETTINGS["popsize"],
        "relative_tolerance": FIT_SETTINGS["relative_tolerance"],
        "absolute_tolerance": FIT_SETTINGS["absolute_tolerance"],
        "cross_over_rate": FIT_SETTINGS["cross_rate"],
        "diff_w": FIT_SETTINGS["diff_w"],
        "init": FIT_SETTINGS["init"],
        "progress_bar": True,
        "display_iterations": True,
        "best_of": RESTART_COUNT,
        "seed": FIT_SETTINGS["seed"],
        "bounds": model["parameters"]["free"],
    }


def validate_fitting_data(data: Mapping[str, Any]) -> np.ndarray:
    """Validate the combined dataset and return its mixed-list trial mask."""

    summary = validate_combined_dataset(data)
    mask = np.asarray(mixed_trial_mask(data), dtype=bool)
    subjects = np.asarray(data["subject"]).reshape(-1)
    mixed_subject_count = int(np.unique(subjects[mask]).size)
    if int(mask.sum()) != MIXED_EXPECTED_LISTS:
        raise RestartError(
            f"Mixed mask selected {int(mask.sum())} lists; "
            f"expected {MIXED_EXPECTED_LISTS}."
        )
    if mixed_subject_count != MIXED_EXPECTED_SUBJECTS:
        raise RestartError(
            f"Mixed mask selected {mixed_subject_count} participants; "
            f"expected {MIXED_EXPECTED_SUBJECTS}."
        )
    if summary["mixed_subject_count"] != MIXED_EXPECTED_SUBJECTS:
        raise RestartError("Combined-dataset summary disagrees with mixed mask.")
    return mask


def _base_metadata(
    task: RestartTask,
    model: Mapping[str, Any],
    *,
    data_sha256: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return immutable metadata attached to one restart result."""

    return {
        "restart_schema_version": SCHEMA_VERSION,
        "task_index": task.task_index,
        "model_index": task.model_index,
        "restart_index": task.restart_index,
        "restart_key": list(task.restart_key),
        "fit_call_index": FIT_CALL_INDEX,
        "requested_best_of": RESTART_COUNT,
        "run_tag": f"{FIT_SETTINGS['base_run_tag']}_restart_{task.restart_index}",
        "data_tag": FIT_SETTINGS["data_tag"],
        "data_path": FIT_SETTINGS["data_path"],
        "data_sha256": data_sha256,
        "trial_query": MIXED_TRIAL_QUERY,
        "model": task.model_name,
        "name": Path(task.filename).stem,
        "components": _json_copy(FIT_SETTINGS["component_paths"]),
        "fit_algorithm": FIT_SETTINGS["fit_alg_path"],
        "loss_function": FIT_SETTINGS["loss_fn_path"],
        "model_factory": model["make_factory_path"],
        "embedding_path": "",
        "emotion_feature_path": "",
        "concat_features": "False",
        "source_sha256": dict(source_sha256),
    }


def fit_task(
    task: RestartTask,
    data: Mapping[str, Any],
    *,
    data_sha256: str,
    source_sha256: Mapping[str, str],
    fitter_cls: type[EvosaxDE] = EvosaxDE,
) -> dict[str, Any]:
    """Execute exactly one fit-only explicit-key restart."""

    model = _model_for_task(task)
    trial_mask = validate_fitting_data(data)
    make_factory = import_from_string(model["make_factory_path"])
    factory = make_factory(
        **{
            name: import_from_string(path)
            for name, path in FIT_SETTINGS["component_paths"].items()
        }
    )
    loss_fn = import_from_string(FIT_SETTINGS["loss_fn_path"])
    fitter = fitter_cls(
        data,
        None,
        model["parameters"]["fixed"],
        factory,
        loss_fn,
        hyperparams=_fitter_hyperparameters(model),
    )
    if not hasattr(fitter, "fit_once"):
        raise RestartError(
            "Configured fitting algorithm does not expose explicit-key fit_once()."
        )

    print(
        f"task={task.task_index}/{TASK_COUNT - 1} model={task.model_name} "
        f"restart={task.restart_index}/{RESTART_COUNT - 1} "
        f"key={list(task.restart_key)}",
        flush=True,
    )
    result = fitter.fit_once(  # type: ignore[attr-defined]
        trial_mask,
        jax.numpy.asarray(task.restart_key, dtype=np.uint32),
        subject_id=-1,
    )
    payload = dict(result)
    payload.update(
        _base_metadata(
            task,
            model,
            data_sha256=data_sha256,
            source_sha256=source_sha256,
        )
    )
    validate_restart_payload(task, payload)
    return payload


def _one_finite_scalar(value: Any, *, label: str) -> float:
    """Return one finite scalar or raise a restart-contract error."""

    array = np.asarray(value)
    if array.size != 1:
        raise RestartError(f"{label} must contain exactly one value.")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise RestartError(f"{label} must be finite; found {scalar}.")
    return scalar


def validate_restart_payload(
    task: RestartTask,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one restart artifact to match its executable task contract."""

    model = _model_for_task(task)
    expected = {
        "restart_schema_version": SCHEMA_VERSION,
        "task_index": task.task_index,
        "model_index": task.model_index,
        "restart_index": task.restart_index,
        "restart_key": list(task.restart_key),
        "fit_call_index": FIT_CALL_INDEX,
        "requested_best_of": RESTART_COUNT,
        "trial_query": MIXED_TRIAL_QUERY,
        "model": task.model_name,
        "name": Path(task.filename).stem,
        "components": _json_copy(FIT_SETTINGS["component_paths"]),
        "fit_algorithm": FIT_SETTINGS["fit_alg_path"],
        "loss_function": FIT_SETTINGS["loss_fn_path"],
        "model_factory": model["make_factory_path"],
        "embedding_path": "",
        "emotion_feature_path": "",
        "concat_features": "False",
    }
    for key, expected_value in expected.items():
        if _json_copy(payload.get(key)) != expected_value:
            raise RestartError(
                f"{task.filename}: expected {key}={expected_value!r}; "
                f"found {payload.get(key)!r}."
            )

    fixed = payload.get("fixed")
    free = payload.get("free")
    fits = payload.get("fits")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(fixed, Mapping):
        raise RestartError(f"{task.filename}: fixed must be a mapping.")
    if not isinstance(free, Mapping):
        raise RestartError(f"{task.filename}: free must be a mapping.")
    if not isinstance(fits, Mapping):
        raise RestartError(f"{task.filename}: fits must be a mapping.")
    if not isinstance(hyperparameters, Mapping):
        raise RestartError(f"{task.filename}: hyperparameters must be a mapping.")

    expected_fixed = {
        name: float(value) for name, value in model["parameters"]["fixed"].items()
    }
    if _json_copy(fixed) != _json_copy(expected_fixed):
        raise RestartError(f"{task.filename}: fixed parameters changed.")
    if _json_copy(free) != _json_copy(model["parameters"]["free"]):
        raise RestartError(f"{task.filename}: free parameter bounds changed.")
    if hyperparameters.get("best_of") != 1:
        raise RestartError(
            f"{task.filename}: one restart must record hyperparameters.best_of=1."
        )

    expected_fit_names = (
        set(expected_fixed) | set(model["parameters"]["free"]) | {"subject"}
    )
    if set(fits) != expected_fit_names:
        raise RestartError(
            f"{task.filename}: unexpected fitted parameter keys "
            f"{sorted(set(fits) ^ expected_fit_names)}."
        )
    for name, value in fits.items():
        scalar = _one_finite_scalar(value, label=f"{task.filename}: fits.{name}")
        if name == "subject" and scalar != -1:
            raise RestartError(f"{task.filename}: pooled subject label must be -1.")

    fitness = _one_finite_scalar(
        payload.get("fitness"),
        label=f"{task.filename}: fitness",
    )
    _one_finite_scalar(payload.get("nit"), label=f"{task.filename}: nit")
    _one_finite_scalar(
        payload.get("fit_time"),
        label=f"{task.filename}: fit_time",
    )
    converged = np.asarray(payload.get("converged"))
    if converged.size != 1:
        raise RestartError(f"{task.filename}: converged must contain one value.")
    if not isinstance(converged.reshape(-1)[0].item(), (bool, np.bool_)):
        raise RestartError(f"{task.filename}: converged must be boolean.")

    data_sha256 = payload.get("data_sha256")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(data_sha256, str) or len(data_sha256) != 64:
        raise RestartError(f"{task.filename}: invalid data_sha256.")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise RestartError(f"{task.filename}: invalid source_sha256.")
    if not all(
        isinstance(value, str) and len(value) == 64 for value in source_sha256.values()
    ):
        raise RestartError(f"{task.filename}: invalid source digest.")

    return {
        "task_index": task.task_index,
        "model": task.model_name,
        "restart_index": task.restart_index,
        "fitness": fitness,
        "nit": int(_one_finite_scalar(payload["nit"], label="nit")),
        "converged": bool(converged.reshape(-1)[0]),
        "fit_time": float(payload["fit_time"]),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON beside its target and atomically install it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_task(
    task_index: int,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load, execute, validate, and atomically save one restart task."""

    task = task_for_index(task_index)
    output_dir = output_dir.resolve()
    output_path = output_dir / task.filename
    if output_path.exists() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        summary = validate_restart_payload(task, payload)
        print(f"Validated existing restart; skipping: {output_path}", flush=True)
        return summary

    data_path = data_path.resolve()
    data = load_data(data_path)
    payload = fit_task(
        task,
        data,
        data_sha256=_sha256(data_path),
        source_sha256=_source_hashes(),
    )
    _atomic_write_json(output_path, payload)
    installed = json.loads(output_path.read_text(encoding="utf-8"))
    summary = validate_restart_payload(task, installed)
    print(f"Installed validated restart: {output_path}", flush=True)
    return summary


def _compatible_restarts(
    model_tasks: Sequence[RestartTask],
    payloads: Sequence[Mapping[str, Any]],
) -> None:
    """Require three restarts to share one scientific specification."""

    compatibility_fields = (
        "data_sha256",
        "source_sha256",
        "trial_query",
        "model",
        "components",
        "fit_algorithm",
        "loss_function",
        "model_factory",
        "fixed",
        "free",
    )
    reference = payloads[0]
    for task, payload in zip(model_tasks, payloads, strict=True):
        validate_restart_payload(task, payload)
        for field in compatibility_fields:
            if _json_copy(payload.get(field)) != _json_copy(reference.get(field)):
                raise RestartError(
                    f"{task.filename}: restart specification differs at {field}."
                )


def reduce_model(
    model_index: int,
    *,
    restart_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one model's minimum-NLL restart and build its canonical payload."""

    if not 0 <= model_index < len(MODEL_COMPARISON_REGISTRY):
        raise RestartError(f"Invalid model_index: {model_index}.")
    model_tasks = restart_tasks()[
        model_index * RESTART_COUNT : (model_index + 1) * RESTART_COUNT
    ]
    payloads = []
    for task in model_tasks:
        path = restart_dir / task.filename
        if not path.is_file():
            raise RestartError(f"Missing restart artifact: {path}.")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    _compatible_restarts(model_tasks, payloads)

    summaries = [
        validate_restart_payload(task, payload)
        for task, payload in zip(model_tasks, payloads, strict=True)
    ]
    winner_position = min(
        range(RESTART_COUNT),
        key=lambda index: (summaries[index]["fitness"], index),
    )
    winner = copy.deepcopy(payloads[winner_position])
    model_name = model_tasks[0].model_name
    winner_task = model_tasks[winner_position]

    for field in (
        "task_index",
        "model_index",
        "restart_index",
        "restart_key",
        "fit_call_index",
        "requested_best_of",
        "restart_schema_version",
    ):
        winner.pop(field, None)
    winner["name"] = f"{model_name}_best_of_{RESTART_COUNT}"
    winner["run_tag"] = f"{FIT_SETTINGS['base_run_tag']}_best_of_{RESTART_COUNT}"
    winner["fit_time"] = float(sum(summary["fit_time"] for summary in summaries))
    winner["hyperparameters"] = dict(winner["hyperparameters"])
    winner["hyperparameters"]["best_of"] = RESTART_COUNT
    winner["restart_selection"] = {
        "winner": winner_task.restart_index,
        "fitness": [summary["fitness"] for summary in summaries],
        "iterations": [summary["nit"] for summary in summaries],
        "converged": [summary["converged"] for summary in summaries],
        "fit_time": [summary["fit_time"] for summary in summaries],
        "source_files": [task.filename for task in model_tasks],
        "selection_rule": "minimum finite fitness, then lowest restart index",
    }
    reduction = {
        "model": model_name,
        "winner": winner_task.restart_index,
        "fitness": winner["fitness"][0],
        "source_files": [task.filename for task in model_tasks],
    }
    return winner, reduction


def reduce_all(
    *,
    restart_dir: Path = DEFAULT_OUTPUT_DIR,
    install: bool = False,
) -> list[dict[str, Any]]:
    """Validate all 48 restarts and optionally install 16 canonical winners."""

    restart_dir = restart_dir.resolve()
    reductions = []
    canonical_payloads = []
    for model_index in range(len(MODEL_COMPARISON_REGISTRY)):
        payload, reduction = reduce_model(model_index, restart_dir=restart_dir)
        canonical_payloads.append(payload)
        reductions.append(reduction)
    if install:
        for payload in canonical_payloads:
            path = restart_dir / f"{payload['name']}.json"
            _atomic_write_json(path, payload)
    return reductions


def preflight(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate the 48-task contract and production prerequisites."""

    tasks = restart_tasks()
    filenames = [task.filename for task in tasks]
    pairs = [(task.model_name, task.restart_index) for task in tasks]
    if len(tasks) != 48 or len(set(filenames)) != 48 or len(set(pairs)) != 48:
        raise RestartError("Restart task grid is not complete and unique.")
    if not hasattr(EvosaxDE, "fit_once"):
        raise RestartError("Installed JAXCMR does not expose EvosaxDE.fit_once().")
    if not output_dir.resolve().is_dir():
        raise RestartError(f"Missing output directory: {output_dir.resolve()}.")

    data_path = data_path.resolve()
    data = load_data(data_path)
    trial_mask = validate_fitting_data(data)
    existing = 0
    for task in tasks:
        path = output_dir / task.filename
        if path.exists():
            validate_restart_payload(
                task,
                json.loads(path.read_text(encoding="utf-8")),
            )
            existing += 1
    return {
        "task_count": len(tasks),
        "model_count": len(MODEL_COMPARISON_REGISTRY),
        "restarts_per_model": RESTART_COUNT,
        "restart_keys": [list(key) for key in _restart_keys()],
        "mixed_lists": int(trial_mask.sum()),
        "mixed_subjects": int(
            np.unique(np.asarray(data["subject"]).reshape(-1)[trial_mask]).size
        ),
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "source_sha256": _source_hashes(),
        "existing_valid_restarts": existing,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Print the deterministic 48-task grid.")

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate data, source, keys, paths, and any existing restarts.",
    )
    preflight_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    preflight_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    fit_parser = subparsers.add_parser(
        "fit",
        help="Execute one explicit-key fit-only restart.",
    )
    fit_parser.add_argument("--task-index", type=int, required=True)
    fit_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    fit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    fit_parser.add_argument("--overwrite", action="store_true")

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="Validate all restarts and select the 16 best-of-three winners.",
    )
    reduce_parser.add_argument(
        "--restart-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    reduce_parser.add_argument(
        "--install",
        action="store_true",
        help="Atomically replace the canonical best-of-three JSONs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the restart worker/reducer command line."""

    args = _parser().parse_args(argv)
    if args.command == "plan":
        for task in restart_tasks():
            print(
                f"{task.task_index:02d} {task.model_name} "
                f"restart={task.restart_index} key={list(task.restart_key)} "
                f"output={task.filename}"
            )
        return 0
    if args.command == "preflight":
        print(
            json.dumps(
                preflight(data_path=args.data_path, output_dir=args.output_dir),
                indent=2,
            )
        )
        return 0
    if args.command == "fit":
        print(
            json.dumps(
                run_task(
                    args.task_index,
                    output_dir=args.output_dir,
                    data_path=args.data_path,
                    overwrite=args.overwrite,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "reduce":
        print(
            json.dumps(
                reduce_all(restart_dir=args.restart_dir, install=args.install),
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
