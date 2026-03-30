"""Single-context model registry for lpp_ecmr render notebooks."""

from .fitting_config import BASE_FREE, ECMR_FREE, EPS, STOP_COMPARISON_ANALYSIS_CONFIGS, STOP_FREE, _NO_LPP_FIXED, _NO_STOP_FIXED

__all__ = ["SINGLE_CONTEXT_MODELS"]

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

_RATIO_STOP_FIELDS = {
    **_POSITIONAL_STOP_FIELDS,
    "component_paths": {
        **_POSITIONAL_STOP_FIELDS["component_paths"],
        "termination_policy_create_fn": "jaxcmr.components.termination.SupportRatioTermination",
    },
}

SINGLE_CONTEXT_MODELS = [
    # ── Enabled models ────────────────────────────────────────────────
    # CMR baseline (k=9)
    {
        "enabled": True,
        "model_name": "CMR",
        "make_factory_path": "jaxcmr.models.cmr.make_factory",
        "parameters": {
            "fixed": _NO_STOP_FIXED,
            "free": {**BASE_FREE},
        },
    },
    # CMR + broad LPP, parsimonious (k=10)
    {
        "enabled": True,
        "model_name": "LPP_CMR",
        "make_factory_path": "lpp_ecmr.models.single_eeg_ecmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **BASE_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
    # CMR baseline + positional termination (k=11)
    {
        "enabled": True,
        "model_name": "CMR_PositionalStop",
        "make_factory_path": "jaxcmr.models.cmr.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
            },
            "free": {**BASE_FREE, **STOP_FREE},
        },
    },
    # CMR + broad LPP, parsimonious + positional termination (k=12)
    {
        "enabled": True,
        "model_name": "LPP_CMR_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.single_eeg_ecmr.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
                "modulate_emotion_by_primacy": False,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **BASE_FREE,
                **STOP_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
    # ── RatioStop ───────────────────────────────────────────────────
    # CMR baseline + ratio termination (k=11)
    {
        "enabled": True,
        "model_name": "CMR_RatioStop",
        "make_factory_path": "jaxcmr.models.cmr.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
            },
            "free": {**BASE_FREE, **STOP_FREE},
        },
    },
    # CMR + broad LPP, parsimonious + ratio termination (k=12)
    {
        "enabled": True,
        "model_name": "LPP_CMR_RatioStop",
        "make_factory_path": "lpp_ecmr.models.single_eeg_ecmr.make_factory",
        **_RATIO_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
                "modulate_emotion_by_primacy": False,
                "emotion_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **BASE_FREE,
                **STOP_FREE,
                "lpp_main_scale": [EPS, 100.0],
            },
        },
    },
]
