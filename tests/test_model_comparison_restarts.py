"""Contracts for independently scheduled pooled optimizer restarts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from lpp_ecmr import model_comparison_restarts as restarts


EXPECTED_KEYS = (
    (4165894930, 804218099),
    (1353695780, 2116000888),
    (2839387376, 2467677468),
)


def _payload(
    task: restarts.RestartTask,
    *,
    fitness: float,
    campaign: str = restarts.DEFAULT_CAMPAIGN,
) -> dict[str, object]:
    model = restarts.MODEL_COMPARISON_REGISTRY[task.model_index]
    fixed = {name: float(value) for name, value in model["parameters"]["fixed"].items()}
    fits = {
        **{name: [value] for name, value in fixed.items()},
        **{name: [0.5] for name in model["parameters"]["free"]},
        "subject": [-1],
    }
    hyperparameters = restarts._fitter_hyperparameters(model, campaign)
    hyperparameters["best_of"] = 1
    return {
        "fixed": fixed,
        "free": copy.deepcopy(model["parameters"]["free"]),
        "fitness": [fitness],
        "fits": fits,
        "nit": [20 + task.restart_index],
        "converged": [True],
        "hyperparameters": hyperparameters,
        "fit_time": 10.0 + task.restart_index,
        **restarts._base_metadata(
            task,
            model,
            data_sha256="d" * 64,
            source_sha256={"source.py": "s" * 64},
            campaign=campaign,
        ),
    }


def test_restart_grid_is_complete_unique_and_uses_historical_keys() -> None:
    tasks = restarts.restart_tasks()

    assert len(tasks) == 48
    assert len({task.filename for task in tasks}) == 48
    assert len({(task.model_name, task.restart_index) for task in tasks}) == 48
    assert tasks[0].model_name == "CMR"
    assert tasks[0].restart_index == 0
    assert tasks[-1].model_name == "EEM_eCMR_LPP_Full"
    assert tasks[-1].restart_index == 2
    assert tuple(task.restart_key for task in tasks[:3]) == EXPECTED_KEYS
    assert set(task.restart_key for task in tasks) == set(EXPECTED_KEYS)

    fit_key = jax.random.fold_in(
        jax.random.PRNGKey(restarts.FIT_SETTINGS["seed"]),
        restarts.FIT_CALL_INDEX,
    )
    expected = tuple(
        tuple(int(word) for word in row)
        for row in np.asarray(jax.random.split(fit_key, 3))
    )
    assert expected == EXPECTED_KEYS


@pytest.mark.parametrize("task_index", [-1, 48])
def test_restart_task_index_rejects_out_of_range(task_index: int) -> None:
    with pytest.raises(restarts.RestartError, match="0..47"):
        restarts.task_for_index(task_index)


def test_restart_validation_rejects_nonfinite_and_wrong_identity() -> None:
    task = restarts.task_for_index(0)
    payload = _payload(task, fitness=10.0)
    restarts.validate_restart_payload(task, payload)

    nonfinite = copy.deepcopy(payload)
    nonfinite["fitness"] = [float("nan")]
    with pytest.raises(restarts.RestartError, match="fitness must be finite"):
        restarts.validate_restart_payload(task, nonfinite)

    wrong_key = copy.deepcopy(payload)
    wrong_key["restart_key"] = [0, 0]
    with pytest.raises(restarts.RestartError, match="restart_key"):
        restarts.validate_restart_payload(task, wrong_key)


def test_stagnation_campaign_is_explicit_and_leaves_default_unchanged() -> None:
    model = restarts.MODEL_COMPARISON_REGISTRY[0]

    standard = restarts._fitter_hyperparameters(model)
    stagnation = restarts._fitter_hyperparameters(
        model,
        restarts.STAGNATION_CAMPAIGN,
    )

    assert standard["num_steps"] == restarts.FIT_SETTINGS["num_steps"]
    assert standard["relative_tolerance"] == restarts.FIT_SETTINGS["relative_tolerance"]
    assert standard["absolute_tolerance"] == restarts.FIT_SETTINGS["absolute_tolerance"]
    assert "stopping_rule" not in standard
    assert stagnation["num_steps"] == 1000
    assert stagnation["stopping_rule"] == "best_nll_stagnation"
    assert stagnation["stagnation_window"] == 100
    assert stagnation["stagnation_tolerance"] == 0.01
    assert "relative_tolerance" not in stagnation
    assert "absolute_tolerance" not in stagnation


def test_restart_validation_keeps_campaigns_separate() -> None:
    task = restarts.task_for_index(0)
    payload = _payload(
        task,
        fitness=10.0,
        campaign=restarts.STAGNATION_CAMPAIGN,
    )

    restarts.validate_restart_payload(
        task,
        payload,
        campaign=restarts.STAGNATION_CAMPAIGN,
    )
    with pytest.raises(restarts.RestartError, match="run_tag"):
        restarts.validate_restart_payload(task, payload)

    wrong_policy = copy.deepcopy(payload)
    wrong_policy["hyperparameters"]["stopping_rule"] = "population_spread"  # type: ignore[index]
    with pytest.raises(restarts.RestartError, match="stopping_rule"):
        restarts.validate_restart_payload(
            task,
            wrong_policy,
            campaign=restarts.STAGNATION_CAMPAIGN,
        )


def test_fit_task_calls_one_explicit_key_restart_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = restarts.task_for_index(0)
    model = restarts.MODEL_COMPARISON_REGISTRY[task.model_index]
    calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    class FakeFitter:
        def __init__(
            self,
            data: object,
            features: object,
            base_params: dict[str, object],
            factory: object,
            loss_fn: object,
            *,
            hyperparams: dict[str, object],
        ) -> None:
            del data, features, factory, loss_fn
            assert hyperparams["best_of"] == 3
            self.base_params = base_params
            self.free = hyperparams["bounds"]

        def fit_once(
            self,
            trial_mask: np.ndarray,
            key: np.ndarray,
            subject_id: int,
        ) -> dict[str, object]:
            calls.append((np.asarray(trial_mask), np.asarray(key), subject_id))
            fixed = {name: float(value) for name, value in self.base_params.items()}
            return {
                "fixed": fixed,
                "free": self.free,
                "fitness": [10.0],
                "fits": {
                    **{name: [value] for name, value in fixed.items()},
                    **{name: [0.5] for name in self.free},
                    "subject": [-1],
                },
                "nit": [5],
                "converged": [True],
                "hyperparameters": {
                    **restarts._fitter_hyperparameters(model),
                    "best_of": 1,
                },
                "fit_time": 2.0,
            }

    def fake_import(path: str) -> object:
        if path == model["make_factory_path"]:
            return lambda **kwargs: ("factory", kwargs)
        return object()

    monkeypatch.setattr(
        restarts,
        "validate_fitting_data",
        lambda data: np.asarray([True, False]),
    )
    monkeypatch.setattr(restarts, "import_from_string", fake_import)

    payload = restarts.fit_task(
        task,
        {"subject": np.asarray([[1], [1]])},
        data_sha256="d" * 64,
        source_sha256={"source.py": "s" * 64},
        fitter_cls=FakeFitter,  # type: ignore[arg-type]
    )

    assert len(calls) == 1
    assert calls[0][0].tolist() == [True, False]
    assert tuple(int(value) for value in calls[0][1]) == task.restart_key
    assert calls[0][2] == -1
    assert payload["name"] == "CMR_restart_0"


def test_reducer_selects_complete_lowest_restart_without_averaging(
    tmp_path: Path,
) -> None:
    tasks = restarts.restart_tasks()[:3]
    fitnesses = [12.0, 8.0, 10.0]
    for task, fitness in zip(tasks, fitnesses, strict=True):
        payload = _payload(task, fitness=fitness)
        payload["fits"]["encoding_drift_rate"] = [0.1 + task.restart_index]  # type: ignore[index]
        (tmp_path / task.filename).write_text(json.dumps(payload), encoding="utf-8")

    winner, summary = restarts.reduce_model(0, restart_dir=tmp_path)

    assert summary["winner"] == 1
    assert winner["fitness"] == [8.0]
    assert winner["fits"]["encoding_drift_rate"] == [1.1]
    assert winner["name"] == "CMR_best_of_3"
    assert winner["hyperparameters"]["best_of"] == 3
    assert winner["fit_time"] == pytest.approx(33.0)
    assert winner["restart_selection"]["winner"] == 1
    assert winner["restart_selection"]["fitness"] == fitnesses


def test_reducer_breaks_fitness_ties_by_restart_index(tmp_path: Path) -> None:
    tasks = restarts.restart_tasks()[:3]
    for task in tasks:
        payload = _payload(task, fitness=8.0)
        (tmp_path / task.filename).write_text(json.dumps(payload), encoding="utf-8")

    winner, summary = restarts.reduce_model(0, restart_dir=tmp_path)

    assert summary["winner"] == 0
    assert winner["restart_selection"]["winner"] == 0


def test_stagnation_reducer_retains_experimental_run_tag(tmp_path: Path) -> None:
    tasks = restarts.restart_tasks()[:3]
    for task in tasks:
        payload = _payload(
            task,
            fitness=8.0 + task.restart_index,
            campaign=restarts.STAGNATION_CAMPAIGN,
        )
        (tmp_path / task.filename).write_text(json.dumps(payload), encoding="utf-8")

    winner, _ = restarts.reduce_model(
        0,
        restart_dir=tmp_path,
        campaign=restarts.STAGNATION_CAMPAIGN,
    )

    assert winner["run_tag"] == (
        "pooled_evosax_set_likelihood_stagnation_w100_tol0p01_best_of_3"
    )
    assert winner["hyperparameters"]["stopping_rule"] == "best_nll_stagnation"


def test_reducer_requires_all_three_compatible_restarts(tmp_path: Path) -> None:
    tasks = restarts.restart_tasks()[:3]
    for task in tasks[:2]:
        (tmp_path / task.filename).write_text(
            json.dumps(_payload(task, fitness=10.0)),
            encoding="utf-8",
        )
    with pytest.raises(restarts.RestartError, match="Missing restart artifact"):
        restarts.reduce_model(0, restart_dir=tmp_path)

    third = _payload(tasks[2], fitness=9.0)
    third["data_sha256"] = "x" * 64
    (tmp_path / tasks[2].filename).write_text(
        json.dumps(third),
        encoding="utf-8",
    )
    with pytest.raises(restarts.RestartError, match="data_sha256"):
        restarts.reduce_model(0, restart_dir=tmp_path)


def _canonical_payloads(
    fitness_by_index: dict[int, float],
) -> tuple[list[dict[str, object]], dict[float, float]]:
    payloads = []
    fitness_by_marker = {}
    for model_index in range(len(restarts.MODEL_COMPARISON_REGISTRY)):
        task = restarts.task_for_index(model_index * restarts.RESTART_COUNT)
        fitness = fitness_by_index[model_index]
        payload = _payload(task, fitness=fitness)
        model = restarts.MODEL_COMPARISON_REGISTRY[model_index]
        for name, bounds in model["parameters"]["free"].items():
            lower, upper = bounds
            payload["fits"][name] = [(lower + upper) / 2]  # type: ignore[index]
        marker = 0.01 * (model_index + 1)
        payload["fits"]["encoding_drift_rate"] = [marker]  # type: ignore[index]
        payload["name"] = f"{task.model_name}_best_of_3"
        payloads.append(payload)
        fitness_by_marker[marker] = fitness
    return payloads, fitness_by_marker


def test_nesting_safeguard_uses_parent_only_when_child_is_worse() -> None:
    fitness_by_index = {index: 1000.0 - index for index in range(16)}
    fitness_by_index[11] = 995.0
    fitness_by_index[15] = 990.0
    payloads, fitness_by_marker = _canonical_payloads(fitness_by_index)
    evaluated_models = []

    def evaluate(model, candidates):
        evaluated_models.append(model["model_name"])
        return [
            fitness_by_marker[float(candidate["encoding_drift_rate"][0])]
            for candidate in candidates
        ]

    selected, audit = restarts.apply_nesting_safeguard(
        payloads,
        evaluate_candidates=evaluate,
    )
    selected_by_model = {payload["model"]: payload for payload in selected}
    audit_by_model = {row["model"]: row for row in audit}

    assert evaluated_models == [
        "CategoryOnly_eCMR_LPP_Full",
        "EEM_eCMR_LPP_Full",
    ]
    category_full = selected_by_model["CategoryOnly_eCMR_LPP_Full"]
    assert category_full["fitness"] == [fitness_by_index[10]]
    assert category_full["nesting_safeguard"]["selected_source"] == (  # type: ignore[index]
        "CategoryOnly_eCMR_LPP_EmotionalOnly"
    )
    assert category_full["fits"]["lpp_main_scale"] == [0.0]  # type: ignore[index]

    eem_full = selected_by_model["EEM_eCMR_LPP_Full"]
    assert eem_full["fitness"] == [fitness_by_index[14]]
    assert audit_by_model["EEM_eCMR_LPP_Full"]["selected_source"] == (
        "EEM_eCMR_LPP_EmotionalOnly"
    )
    assert eem_full["fits"]["lpp_main_scale"] == [0.0]  # type: ignore[index]


def test_nesting_safeguard_skips_objective_when_all_children_beat_parents() -> None:
    payloads, _ = _canonical_payloads({index: 1000.0 - index for index in range(16)})

    def unexpected_evaluation(model, candidates):
        del model, candidates
        raise AssertionError("No parent fallback should require reevaluation.")

    selected, audit = restarts.apply_nesting_safeguard(
        payloads,
        evaluate_candidates=unexpected_evaluation,
    )

    assert [payload["fitness"] for payload in selected] == [
        [1000.0 - index] for index in range(16)
    ]
    assert {row["selected_source"] for row in audit} == {"optimizer"}
