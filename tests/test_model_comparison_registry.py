from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from jax import numpy as jnp
from jaxcmr.components.termination import NoStopTermination

from lpp_ecmr.data_contract import (
    MIXED_EXPECTED_LISTS,
    MIXED_EXPECTED_SUBJECTS,
    MIXED_TRIAL_QUERY,
)
from lpp_ecmr.model_comparison_registry import (
    EXPECTED_MODEL_NAMES,
    FIT_SETTINGS,
    LPP_CENTERED_MAX,
    LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT,
    LPP_SLOPE_BOUNDS,
    MODEL_COMPARISON_REGISTRY,
    NESTING_RELATIONSHIPS,
    PARAMETER_NEUTRAL_VALUES,
    SOURCE_ENCODING_DRIFT_BOUNDS,
    SOURCE_LEARNING_BOUNDS,
)
from lpp_ecmr.model_comparison_workflow import (
    _make_notebook_query_cohort_aware,
)
from lpp_ecmr.models.full_eeg_ecmr import eCMR
from lpp_ecmr.models.learning_strength import compose_learning_strength
from lpp_ecmr.models.single_eeg_ecmr import CMR


def _lookup():
    return {model["model_name"]: model for model in MODEL_COMPARISON_REGISTRY}


def test_registry_is_complete_and_ordered():
    assert tuple(model["model_name"] for model in MODEL_COMPARISON_REGISTRY) == (
        EXPECTED_MODEL_NAMES
    )
    assert len({model["model_name"] for model in MODEL_COMPARISON_REGISTRY}) == 16


def test_registry_free_parameter_counts():
    counts = {
        model["model_name"]: len(model["parameters"]["free"])
        for model in MODEL_COMPARISON_REGISTRY
    }
    assert list(counts.values()) == [
        9,
        10,
        10,
        11,
        10,
        11,
        11,
        12,
        11,
        12,
        12,
        13,
        12,
        13,
        13,
        14,
    ]


def test_all_models_share_frozen_fitting_policy():
    assert FIT_SETTINGS["pooled"] is True
    assert FIT_SETTINGS["trial_query"] == MIXED_TRIAL_QUERY
    assert FIT_SETTINGS["expected_list_count"] == MIXED_EXPECTED_LISTS == 342
    assert MIXED_EXPECTED_SUBJECTS == 38
    assert FIT_SETTINGS["best_of"] == 3
    assert FIT_SETTINGS["num_steps"] == 1000
    assert FIT_SETTINGS["learning_strength_link"] == "log"
    assert FIT_SETTINGS["lpp_slope_direction"] == "nonnegative"
    assert "ExcludeTermination" in FIT_SETTINGS["loss_fn_path"]
    assert "NoStopTermination" in str(FIT_SETTINGS["component_paths"])
    for model in MODEL_COMPARISON_REGISTRY:
        fixed = model["parameters"]["fixed"]
        assert fixed["learn_after_context_update"] is True
        assert fixed["allow_repeated_recalls"] is False


def test_generated_fit_notebook_applies_subject_limit_within_trial_query():
    notebook = SimpleNamespace(
        cells=[
            {
                "cell_type": "code",
                "metadata": {"tags": ["parameters"]},
                "source": (
                    "trial_query = \"data['listtype'] == -1\"\n"
                    "data = load_data(os.path.join(project_root, data_path), "
                    "max_subjects)\n"
                    "trial_mask = generate_trial_mask(data, trial_query)\n"
                    'unique_subjects = jnp.unique(jnp.array(data["subject"]))\n'
                ),
            }
        ]
    )

    _make_notebook_query_cohort_aware(notebook)

    source = notebook.cells[0]["source"]
    assert "load_data(os.path.join(project_root, data_path), 0)" in source
    assert 'reshape(-1)[trial_mask]' in source
    assert "cohort_subjects" in source
    assert "max_subjects)" not in source
    assert f"trial_query = {MIXED_TRIAL_QUERY!r}" in source
    assert "listtype" not in source


def test_source_models_share_architectural_parameters_and_drift_policy():
    for model in MODEL_COMPARISON_REGISTRY:
        fixed = model["parameters"]["fixed"]
        free = model["parameters"]["free"]
        if model["architecture"] == "source":
            assert free["source_learning_rate"] == SOURCE_LEARNING_BOUNDS
            assert free["source_encoding_drift_rate"] == SOURCE_ENCODING_DRIFT_BOUNDS
            assert fixed["source_recall_drift_rate"] == 1.0
            assert "emotion_drift_rate" not in fixed | free
        else:
            assert "source_learning_rate" not in fixed | free
            assert "source_encoding_drift_rate" not in fixed | free
            assert "source_recall_drift_rate" not in fixed | free


def test_lpp_slopes_use_approved_log_scale_bounds():
    assert LPP_SLOPE_BOUNDS[0] == 0.0
    assert LPP_SLOPE_BOUNDS[1] < 1.0
    assert np.isclose(
        np.exp(LPP_SLOPE_BOUNDS[1] * LPP_CENTERED_MAX),
        LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT,
    )
    for model in MODEL_COMPARISON_REGISTRY:
        free = model["parameters"]["free"]
        for parameter in ("lpp_main_scale", "lpp_inter_scale"):
            if parameter in free:
                assert free[parameter] == LPP_SLOPE_BOUNDS


def test_every_nesting_edge_adds_exactly_one_free_parameter():
    models = _lookup()
    for edge in NESTING_RELATIONSHIPS:
        parent = models[edge["parent"]]["parameters"]
        child = models[edge["child"]]["parameters"]
        added = edge["added_parameter"]
        assert set(child["free"]) == set(parent["free"]) | {added}
        assert parent["fixed"][added] == PARAMETER_NEUTRAL_VALUES[added]
        child_fixed_without_added = deepcopy(child["fixed"])
        parent_fixed_without_added = deepcopy(parent["fixed"])
        parent_fixed_without_added.pop(added)
        assert child_fixed_without_added == parent_fixed_without_added


def test_categorical_multiplier_preserves_neutral_and_scales_emotional():
    neutral = float(compose_learning_strength(2.0, 0.0, 0.5, 1.7, 0.3, 0.2))
    emotional = float(compose_learning_strength(2.0, 1.0, 0.5, 1.7, 0.3, 0.2))
    disabled = float(compose_learning_strength(2.0, 1.0, 0.5, 1.0, 0.3, 0.2))

    assert np.isclose(neutral, 2.0 * np.exp(0.3 * 0.5))
    assert np.isclose(emotional, 2.0 * 1.7 * np.exp((0.3 + 0.2) * 0.5))
    assert np.isclose(disabled, 2.0 * np.exp((0.3 + 0.2) * 0.5))


def test_log_lpp_terms_have_independent_zero_values_and_expected_slopes():
    baseline = 2.0
    lpp = 0.5
    emotion_scale = 1.7
    main = 0.3
    interaction = 0.2
    full = float(
        compose_learning_strength(
            baseline, 1.0, lpp, emotion_scale, main, interaction
        )
    )
    no_main = float(
        compose_learning_strength(
            baseline, 1.0, lpp, emotion_scale, 0.0, interaction
        )
    )
    no_interaction = float(
        compose_learning_strength(baseline, 1.0, lpp, emotion_scale, main, 0.0)
    )

    assert np.isclose(np.log(full / no_main), main * lpp)
    assert np.isclose(np.log(full / no_interaction), interaction * lpp)


def test_completed_learning_strength_is_positive_without_clamping():
    completed = compose_learning_strength(0.25, 0.0, -1.0, 1.0, 1.0, 0.0)
    assert float(completed) > 0
    assert np.isclose(float(completed), 0.25 * np.exp(-1.0))


def _source_parameters():
    return {
        "encoding_drift_rate": 0.6,
        "start_drift_rate": 0.8,
        "recall_drift_rate": 0.7,
        "source_encoding_drift_rate": 0.4,
        "source_recall_drift_rate": 1.0,
        "shared_support": 0.1,
        "item_support": 0.2,
        "learning_rate": 0.5,
        "primacy_scale": 0.1,
        "primacy_decay": 1.0,
        "choice_sensitivity": 10.0,
        "source_learning_rate": 0.3,
        "emotion_scale": 1.0,
        "lpp_main_scale": 0.0,
        "lpp_inter_scale": 0.0,
        "allow_repeated_recalls": False,
        "learn_after_context_update": True,
        "phi_emot_modulates_temporal": False,
    }


@pytest.mark.parametrize(
    ("model_type", "learning_method", "baseline_scale"),
    (
        (CMR, "_mcf_learning_rate", 1.0),
        (eCMR, "_source_mcf_learning_rate", 0.3),
    ),
)
def test_models_apply_categorical_multiplier_to_ordinary_baseline(
    model_type, learning_method, baseline_scale
):
    parameters = _source_parameters() | {"emotion_scale": 1.7}
    models = {
        emotional: model_type(
            2,
            parameters,
            jnp.asarray([emotional, not emotional]),
            jnp.zeros(2),
            termination_policy_create_fn=NoStopTermination,
        )
        for emotional in (False, True)
    }
    baseline = float(models[False].primacy[0]) * baseline_scale

    assert np.isclose(float(getattr(models[False], learning_method)()), baseline)
    assert np.isclose(
        float(getattr(models[True], learning_method)()),
        baseline * parameters["emotion_scale"],
    )


def test_source_learning_scale_only_multiplies_ordinary_baseline():
    parameters = _source_parameters()
    model = eCMR(
        2,
        parameters,
        jnp.asarray([True, False]),
        jnp.zeros(2),
        termination_policy_create_fn=NoStopTermination,
    )
    primacy = float(model.primacy[0])
    assert np.isclose(
        float(model._source_mcf_learning_rate()),
        parameters["source_learning_rate"] * primacy,
    )

    equal_baseline_model = eCMR(
        2,
        parameters | {"source_learning_rate": 1.0},
        jnp.asarray([True, False]),
        jnp.zeros(2),
        termination_policy_create_fn=NoStopTermination,
    )
    assert np.isclose(
        float(equal_baseline_model._source_mcf_learning_rate()),
        float(equal_baseline_model._temporal_mcf_learning_rate()),
    )


@pytest.mark.parametrize(
    "emotions",
    ([True, True], [True, False], [False, True], [False, False]),
    ids=(
        "emotional-emotional",
        "emotional-neutral",
        "neutral-emotional",
        "neutral-neutral",
    ),
)
def test_source_associations_use_each_items_updated_context(emotions):
    model = eCMR(
        2,
        _source_parameters(),
        jnp.asarray(emotions),
        jnp.zeros(2),
        termination_policy_create_fn=NoStopTermination,
    )

    for item_index in range(2):
        item = model.items[item_index]
        updated_context = model.emotion_context.integrate(
            model.emotion_mfc.probe(item), model.source_encoding_drift_rate
        )
        learning_rate = model._source_mcf_learning_rate()
        old_column = model.emotion_mcf.state[:, item_index]

        model = model.experience_item(item_index)

        learned_column = model.emotion_mcf.state[:, item_index] - old_column
        np.testing.assert_allclose(
            learned_column,
            learning_rate * updated_context.state,
            atol=1e-6,
        )


def test_source_recall_drift_is_separate_from_temporal_recall_drift():
    parameters = _source_parameters() | {
        "recall_drift_rate": 0.3,
        "source_recall_drift_rate": 1.0,
    }
    model = eCMR(
        2,
        parameters,
        jnp.asarray([True, False]),
        jnp.zeros(2),
        termination_policy_create_fn=NoStopTermination,
    )
    item = model.items[0]
    expected_temporal = model.context.integrate(
        model.mfc.probe(item), parameters["recall_drift_rate"]
    )
    expected_source = model.emotion_context.integrate(
        model.emotion_mfc.probe(item), parameters["source_recall_drift_rate"]
    )

    recalled = model._retrieve_item(0)

    np.testing.assert_allclose(recalled.context.state, expected_temporal.state)
    np.testing.assert_allclose(recalled.emotion_context.state, expected_source.state)
