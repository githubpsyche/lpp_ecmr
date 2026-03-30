"""Strength-only model registry for lpp_ecmr render notebooks.

No-context baselines that test whether CMR's context machinery
improves over encoding strength + Luce choice alone.
"""

from .fitting_config import EPS, STOP_COMPARISON_ANALYSIS_CONFIGS, STOP_FREE, _NO_LPP_FIXED

__all__ = ["STRENGTH_MODELS"]

_POSITIONAL_STOP_FIELDS = {
    "base_run_tag": "50_set_likelihood_predicted_term",
    "component_paths": {
        "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
        "context_create_fn": "jaxcmr.components.context.init",
        "termination_policy_create_fn": "jaxcmr.components.termination.PositionalTermination",
    },
    "sim_alg_path": "jaxcmr.simulation.simulate_study_and_free_recall",
    "loss_fn_path": "jaxcmr.loss.set_permutation_likelihood.IncludeTerminationLikelihoodFnGenerator",
    "comparison_analysis_configs": STOP_COMPARISON_ANALYSIS_CONFIGS,
}

_STRENGTH_FREE = {
    "primacy_scale": [EPS, 100.0],
    "primacy_decay": [EPS, 100.0],
    "choice_sensitivity": [EPS, 100.0],
}

_STRENGTH_FIXED = {
    "allow_repeated_recalls": False,
    "modulate_emotion_by_primacy": False,
}

_RATIO_STOP_FIELDS = {
    **_POSITIONAL_STOP_FIELDS,
    "component_paths": {
        **_POSITIONAL_STOP_FIELDS["component_paths"],
        "termination_policy_create_fn": "jaxcmr.components.termination.SupportRatioTermination",
    },
}

STRENGTH_MODELS = [
    # ── No stop ───────────────────────────────────────────────────────
    # Strength baseline (k=3)
    {
        "enabled": True,
        "model_name": "Strength",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                **_NO_LPP_FIXED,
            },
            "free": {**_STRENGTH_FREE},
        },
    },
    # Strength + broad LPP, parsimonious (k=4)
    {
        "enabled": True,
        "model_name": "LPP_Strength",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
    # Strength + emotion (k=4)
    {
        "enabled": True,
        "model_name": "Strength_Emotion",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                **_NO_LPP_FIXED,
            },
            "free": {
                **_STRENGTH_FREE,
                "emotion_scale": [EPS, 10.0],
            },
        },
    },
    # Strength + emotion + emotion-gated LPP, parsimonious (k=5)
    {
        "enabled": True,
        "model_name": "LPP_Strength_Emotion",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                "emotion_scale": [EPS, 10.0],
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
    # ── PositionalStop ────────────────────────────────────────────────
    # Strength baseline + positional termination (k=5)
    {
        "enabled": True,
        "model_name": "Strength_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                **_NO_LPP_FIXED,
            },
            "free": {**_STRENGTH_FREE, **STOP_FREE},
        },
    },
    # Strength + broad LPP, parsimonious + positional termination (k=6)
    {
        "enabled": True,
        "model_name": "LPP_Strength_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
    # Strength + emotion + positional termination (k=6)
    {
        "enabled": True,
        "model_name": "Strength_Emotion_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                **_NO_LPP_FIXED,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "emotion_scale": [EPS, 10.0],
            },
        },
    },
    # Strength + emotion + emotion-gated LPP, parsimonious + positional termination (k=7)
    {
        "enabled": True,
        "model_name": "LPP_Strength_Emotion_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "emotion_scale": [EPS, 10.0],
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
    # ── RatioStop ──────────────────────────────────────────────────
    # Strength baseline + ratio termination (k=5)
    {
        "enabled": True,
        "model_name": "Strength_RatioStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                **_NO_LPP_FIXED,
            },
            "free": {**_STRENGTH_FREE, **STOP_FREE},
        },
    },
    # Strength + broad LPP, parsimonious + ratio termination (k=6)
    {
        "enabled": True,
        "model_name": "LPP_Strength_RatioStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
    # Strength + emotion + ratio termination (k=6)
    {
        "enabled": True,
        "model_name": "Strength_Emotion_RatioStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                **_NO_LPP_FIXED,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "emotion_scale": [EPS, 10.0],
            },
        },
    },
    # Strength + emotion + emotion-gated LPP, parsimonious + ratio termination (k=7)
    {
        "enabled": True,
        "model_name": "LPP_Strength_Emotion_RatioStop",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                **_STRENGTH_FIXED,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **_STRENGTH_FREE,
                **STOP_FREE,
                "emotion_scale": [EPS, 10.0],
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
]
