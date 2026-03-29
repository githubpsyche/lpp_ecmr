"""Full eCMR model registry for lpp_ecmr render notebooks."""

from .fitting_config import ECMR_FREE, EPS, STOP_FREE, _NO_LPP_FIXED, _NO_STOP_FIXED

__all__ = ["ECMR_MODELS"]

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
}

ECMR_MODELS = [
    # ── Enabled models ────────────────────────────────────────────────
    # eCMR emotion only, no LPP (k=10)
    {
        "enabled": True,
        "model_name": "eCMR",
        "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
                **_NO_LPP_FIXED,
            },
            "free": {**ECMR_FREE},
        },
    },
    # eCMR + emotion-gated LPP, parsimonious (k=11)
    {
        "enabled": True,
        "model_name": "LPP_eCMR",
        "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **ECMR_FREE,
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
    # eCMR emotion only + positional termination (k=12)
    {
        "enabled": True,
        "model_name": "eCMR_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_LPP_FIXED,
            },
            "free": {**ECMR_FREE, **STOP_FREE},
        },
    },
    # eCMR + emotion-gated LPP, parsimonious + positional termination (k=13)
    {
        "enabled": True,
        "model_name": "LPP_eCMR_PositionalStop",
        "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
        **_POSITIONAL_STOP_FIELDS,
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "learn_after_context_update": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **ECMR_FREE,
                **STOP_FREE,
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
    # ── Commented-out variants ────────────────────────────────────────
    # # eCMR Broad — phi_emot modulates both temporal and emotional MCF
    # {
    #     "enabled": False,
    #     "model_name": "eCMR_Broad",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             **_NO_STOP_FIXED,
    #             **_NO_LPP_FIXED,
    #             "phi_emot_modulates_temporal": True,
    #         },
    #         "free": {**ECMR_FREE},
    #     },
    # },
    # # LPP-eCMR Broad — emotion-gated LPP + phi_emot on both pathways
    # {
    #     "enabled": False,
    #     "model_name": "LPP_eCMR_Broad",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             **_NO_STOP_FIXED,
    #             "lpp_main_scale": 0.0,
    #             "lpp_main_threshold": 0.0,
    #             "lpp_inter_threshold": 0.0,
    #             "phi_emot_modulates_temporal": True,
    #         },
    #         "free": {
    #             **ECMR_FREE,
    #             "lpp_inter_scale": [EPS, 100.0],
    #         },
    #     },
    # },
    # # eCMR + general LPP via phi_emot (main effect on all items, k=12)
    # {
    #     "enabled": False,
    #     "model_name": "eCMRMainEffects",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             **_NO_STOP_FIXED,
    #             "lpp_inter_scale": 0.0,
    #             "lpp_inter_threshold": 0.0,
    #         },
    #         "free": {
    #             **ECMR_FREE,
    #             "lpp_main_scale": [EPS, 100.0],
    #             "lpp_main_threshold": [-5.0, 5.0],
    #         },
    #     },
    # },
    # # eCMR + general LPP + interaction (k=14)
    # {
    #     "enabled": False,
    #     "model_name": "eCMRInteraction",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             **_NO_STOP_FIXED,
    #         },
    #         "free": {
    #             **ECMR_FREE,
    #             "lpp_main_scale": [EPS, 100.0],
    #             "lpp_main_threshold": [-5.0, 5.0],
    #             "lpp_inter_scale": [EPS, 100.0],
    #             "lpp_inter_threshold": [-5.0, 5.0],
    #         },
    #     },
    # },
    # # eCMR Broad + positional termination
    # {
    #     "enabled": False,
    #     "model_name": "eCMR_Broad_PositionalStop",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     **_POSITIONAL_STOP_FIELDS,
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "learn_after_context_update": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             **_NO_LPP_FIXED,
    #             "phi_emot_modulates_temporal": True,
    #         },
    #         "free": {**ECMR_FREE, **STOP_FREE},
    #     },
    # },
    # # LPP-eCMR Broad + positional termination
    # {
    #     "enabled": False,
    #     "model_name": "LPP_eCMR_Broad_PositionalStop",
    #     "make_factory_path": "lpp_ecmr.models.full_eeg_ecmr.make_factory",
    #     **_POSITIONAL_STOP_FIELDS,
    #     "parameters": {
    #         "fixed": {
    #             "allow_repeated_recalls": False,
    #             "learn_after_context_update": False,
    #             "modulate_emotion_by_primacy": False,
    #             "emotion_drift_rate": 1.0,
    #             "lpp_main_scale": 0.0,
    #             "lpp_main_threshold": 0.0,
    #             "lpp_inter_threshold": 0.0,
    #             "phi_emot_modulates_temporal": True,
    #         },
    #         "free": {
    #             **ECMR_FREE,
    #             **STOP_FREE,
    #             "lpp_inter_scale": [EPS, 100.0],
    #         },
    #     },
    # },
]
