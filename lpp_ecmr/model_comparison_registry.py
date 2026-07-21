"""Canonical pooled 16-model comparison specified in GitHub issue #2."""

from __future__ import annotations

from copy import deepcopy
from math import log
from typing import Any

from .data_contract import MIXED_EXPECTED_LISTS, MIXED_TRIAL_QUERY
from .fitting_config import BASE_FREE, EPS

__all__ = [
    "EXPECTED_MODEL_NAMES",
    "FIT_SETTINGS",
    "LPP_CENTERED_MAX",
    "LPP_CENTERED_MIN",
    "LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT",
    "LPP_SLOPE_BOUNDS",
    "MODEL_COMPARISON_REGISTRY",
    "NESTING_RELATIONSHIPS",
    "PARAMETER_MANUSCRIPT_NAMES",
    "PARAMETER_NEUTRAL_VALUES",
    "SOURCE_ENCODING_DRIFT_BOUNDS",
    "SOURCE_LEARNING_BOUNDS",
    "build_model_comparison_registry",
]


# Observed range after the EarlyLPP values in TalmiEEG.h5 are centered within
# each 20-item study list. The log-slope ceiling lets either LPP term contribute
# at most a tenfold multiplier at the largest observed positive value. In the
# full model, both terms can therefore contribute at most 10 x 10 = 100 for an
# emotional item. The directional hypothesis is nonnegative (index.qmd reports
# positive Early-LPP main and emotion x Early-LPP effects).
LPP_CENTERED_MIN = -8.3405953
LPP_CENTERED_MAX = 10.732445846153846
LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT = 10.0
LPP_SLOPE_BOUNDS = [
    0.0,
    log(LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT) / LPP_CENTERED_MAX,
]
SOURCE_LEARNING_BOUNDS = [EPS, 10.0]
SOURCE_ENCODING_DRIFT_BOUNDS = list(BASE_FREE["encoding_drift_rate"])

EXPECTED_MODEL_NAMES = (
    "CMR",
    "CMR_LPP_General",
    "CMR_LPP_EmotionalOnly",
    "CMR_LPP_Full",
    "EEM_CMR",
    "EEM_CMR_LPP_General",
    "EEM_CMR_LPP_EmotionalOnly",
    "EEM_CMR_LPP_Full",
    "CategoryOnly_eCMR",
    "CategoryOnly_eCMR_LPP_General",
    "CategoryOnly_eCMR_LPP_EmotionalOnly",
    "CategoryOnly_eCMR_LPP_Full",
    "EEM_eCMR",
    "EEM_eCMR_LPP_General",
    "EEM_eCMR_LPP_EmotionalOnly",
    "EEM_eCMR_LPP_Full",
)

FIT_SETTINGS: dict[str, Any] = {
    "data_tag": "TalmiEEG",
    "data_path": "data/TalmiEEG.h5",
    "trial_query": MIXED_TRIAL_QUERY,
    "expected_list_count": MIXED_EXPECTED_LISTS,
    "pooled": True,
    "base_run_tag": "pooled_evosax_set_likelihood",
    "fit_alg_path": "jaxcmr.fitting.EvosaxDE",
    "loss_fn_path": (
        "jaxcmr.loss.set_permutation_likelihood.ExcludeTerminationLikelihoodLoss"
    ),
    "sim_alg_path": ("jaxcmr.simulation.simulate_study_free_recall_and_forced_stop"),
    "component_paths": {
        "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
        "context_create_fn": "jaxcmr.components.context.init",
        "termination_policy_create_fn": (
            "jaxcmr.components.termination.NoStopTermination"
        ),
    },
    "seed": 0,
    "relative_tolerance": 0.0001,
    "absolute_tolerance": 0.0,
    "popsize": 15,
    "num_steps": 1000,
    "cross_rate": 0.9,
    "diff_w": (0.5, 1.0),
    "init": "latinhypercube",
    "best_of": 3,
    "experiment_count": 200,
    "lpp_preprocessing": "within-list mean centering of EarlyLPP",
    "learning_strength_link": "log",
    "lpp_slope_direction": "nonnegative",
    "lpp_observed_centered_range": [LPP_CENTERED_MIN, LPP_CENTERED_MAX],
    "lpp_single_effect_multiplier_limit": LPP_SINGLE_EFFECT_MULTIPLIER_LIMIT,
    "source_learning_bounds_rationale": (
        "source_learning_rate is a positive scale relative to temporal "
        "learning; 1 gives equal ordinary strength and the upper bound allows "
        "source learning up to one order of magnitude stronger"
    ),
}

_EMOTION_FREE = [EPS, 10.0]

PARAMETER_NEUTRAL_VALUES = {
    "emotion_scale": 1.0,
    "lpp_main_scale": 0.0,
    "lpp_inter_scale": 0.0,
}

PARAMETER_MANUSCRIPT_NAMES = {
    "source_encoding_drift_rate": "beta_enc_source",
    "source_recall_drift_rate": "beta_rec_source",
    "source_learning_rate": "omega",
    "emotion_scale": "phi_emot",
    "lpp_main_scale": "kappa",
    "lpp_inter_scale": "kappa_emot",
}

_LPP_LABELS = {
    (False, False): "none",
    (True, False): "general",
    (False, True): "emotional_only",
    (True, True): "full",
}


def _model_name(
    architecture: str,
    categorical_enhancement: bool,
    general_lpp: bool,
    emotional_lpp: bool,
) -> str:
    lpp = _LPP_LABELS[(general_lpp, emotional_lpp)]
    if architecture == "temporal":
        prefix = "EEM_CMR" if categorical_enhancement else "CMR"
    else:
        prefix = "EEM_eCMR" if categorical_enhancement else "CategoryOnly_eCMR"

    suffix = {
        "none": "",
        "general": "_LPP_General",
        "emotional_only": "_LPP_EmotionalOnly",
        "full": "_LPP_Full",
    }[lpp]
    return prefix + suffix


def _parameters(
    architecture: str,
    categorical_enhancement: bool,
    general_lpp: bool,
    emotional_lpp: bool,
) -> dict[str, dict[str, Any]]:
    fixed: dict[str, Any] = {
        "allow_repeated_recalls": False,
        "learn_after_context_update": True,
    }
    free = deepcopy(BASE_FREE)

    for enabled, name, bounds in (
        (categorical_enhancement, "emotion_scale", _EMOTION_FREE),
        (general_lpp, "lpp_main_scale", LPP_SLOPE_BOUNDS),
        (emotional_lpp, "lpp_inter_scale", LPP_SLOPE_BOUNDS),
    ):
        if enabled:
            free[name] = list(bounds)
        else:
            fixed[name] = PARAMETER_NEUTRAL_VALUES[name]

    if architecture == "source":
        fixed |= {
            "source_recall_drift_rate": 1.0,
            "phi_emot_modulates_temporal": False,
        }
        free["source_encoding_drift_rate"] = list(SOURCE_ENCODING_DRIFT_BOUNDS)
        free["source_learning_rate"] = list(SOURCE_LEARNING_BOUNDS)

    return {"fixed": fixed, "free": free}


def build_model_comparison_registry() -> list[dict[str, Any]]:
    """Build the complete factorial model registry in issue order."""

    registry = []
    for architecture in ("temporal", "source"):
        for categorical_enhancement in (False, True):
            for general_lpp, emotional_lpp in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                registry.append(
                    {
                        "model_name": _model_name(
                            architecture,
                            categorical_enhancement,
                            general_lpp,
                            emotional_lpp,
                        ),
                        "architecture": architecture,
                        "categorical_enhancement": categorical_enhancement,
                        "general_lpp": general_lpp,
                        "emotional_lpp": emotional_lpp,
                        "lpp_form": _LPP_LABELS[(general_lpp, emotional_lpp)],
                        "make_factory_path": (
                            "lpp_ecmr.models.single_eeg_ecmr.make_factory"
                            if architecture == "temporal"
                            else "lpp_ecmr.models.full_eeg_ecmr.make_factory"
                        ),
                        "parameters": _parameters(
                            architecture,
                            categorical_enhancement,
                            general_lpp,
                            emotional_lpp,
                        ),
                    }
                )
    return registry


MODEL_COMPARISON_REGISTRY = build_model_comparison_registry()


def _build_nesting_relationships() -> tuple[dict[str, str], ...]:
    lookup = {
        (
            model["architecture"],
            model["categorical_enhancement"],
            model["general_lpp"],
            model["emotional_lpp"],
        ): model["model_name"]
        for model in MODEL_COMPARISON_REGISTRY
    }
    edges = []
    factors = (
        ("categorical_enhancement", 1, "emotion_scale"),
        ("general_lpp", 2, "lpp_main_scale"),
        ("emotional_lpp", 3, "lpp_inter_scale"),
    )
    for key, child in lookup.items():
        for factor, index, parameter in factors:
            if not key[index]:
                continue
            parent_key = list(key)
            parent_key[index] = False
            edges.append(
                {
                    "parent": lookup[tuple(parent_key)],
                    "child": child,
                    "factor": factor,
                    "added_parameter": parameter,
                }
            )
    return tuple(edges)


NESTING_RELATIONSHIPS = _build_nesting_relationships()


assert tuple(model["model_name"] for model in MODEL_COMPARISON_REGISTRY) == (
    EXPECTED_MODEL_NAMES
)
assert len(MODEL_COMPARISON_REGISTRY) == 16
assert len(NESTING_RELATIONSHIPS) == 24
