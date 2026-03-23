"""Full eCMR model registry for lpp_ecmr render notebooks."""

from .fitting_config import ECMR_FREE, EPS, _NO_LPP_FIXED, _NO_STOP_FIXED

__all__ = ["ECMR_MODELS"]


ECMR_MODELS = [
    # eCMR Emotion-only (no LPP terms)
    {
        "enabled": False,
        "model_name": "eCMREmotionOnly",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
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
    # eCMR + LPP main effect
    {
        "enabled": False,
        "model_name": "eCMRMainEffects",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **ECMR_FREE,
                "lpp_main_scale": [EPS, 100.0],
                "lpp_main_threshold": [-5.0, 5.0],
            },
        },
    },
    # eCMR + LPP main + interaction
    {
        "enabled": False,
        "model_name": "eCMRInteraction",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
            },
            "free": {
                **ECMR_FREE,
                "lpp_main_scale": [EPS, 100.0],
                "lpp_main_threshold": [-5.0, 5.0],
                "lpp_inter_scale": [EPS, 100.0],
                "lpp_inter_threshold": [-5.0, 5.0],
            },
        },
    },
    # eCMR Emo x LPP - parsimonious: 1 LPP param, emotional items only, no threshold
    {
        "enabled": False,
        "model_name": "eCMREmotionTimesLPP",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
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
    # eCMR Emo x LPP Broad - phi_emot modulates both temporal and emotional MCF
    {
        "enabled": True,
        "model_name": "eCMREmotionTimesLPPBroad",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
                "lpp_main_scale": 0.0,
                "lpp_main_threshold": 0.0,
                "lpp_inter_threshold": 0.0,
                "phi_emot_modulates_temporal": True,
            },
            "free": {
                **ECMR_FREE,
                "lpp_inter_scale": [EPS, 100.0],
            },
        },
    },
    # eCMR Emotion Broad - phi_emot on both pathways, no LPP (matched baseline)
    {
        "enabled": True,
        "model_name": "eCMREmotionBroad",
        "make_factory_path": "lpp_ecmr.models.eeg_full_ecmr.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
                "emotion_drift_rate": 1.0,
                **_NO_STOP_FIXED,
                **_NO_LPP_FIXED,
                "phi_emot_modulates_temporal": True,
            },
            "free": {**ECMR_FREE},
        },
    },
]
