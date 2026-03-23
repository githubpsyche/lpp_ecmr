"""Single-context model registry for lpp_ecmr render notebooks."""

from .fitting_config import BASE_FREE, ECMR_FREE, EPS, _NO_LPP_FIXED, _NO_STOP_FIXED

__all__ = ["SINGLE_CONTEXT_MODELS"]


SINGLE_CONTEXT_MODELS = [
    # Strength-only EEG model (no context dynamics)
    {
        "enabled": False,
        "model_name": "EEGStrength",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": False,
            },
            "free": {
                "primacy_scale": [EPS, 100.0],
                "primacy_decay": [EPS, 100.0],
                "choice_sensitivity": [EPS, 100.0],
                "lpp_threshold": [-5.0, 5.0],
                "lpp_slope": [EPS, 10.0],
            },
        },
    },
    # Strength-only EEG model; multiplicative emotion-primacy rule
    {
        "enabled": False,
        "model_name": "EEGStrengthMultiplicative",
        "make_factory_path": "lpp_ecmr.models.eeg_strength.make_factory",
        "parameters": {
            "fixed": {
                "allow_repeated_recalls": False,
                "modulate_emotion_by_primacy": True,
            },
            "free": {
                "primacy_scale": [EPS, 100.0],
                "primacy_decay": [EPS, 100.0],
                "choice_sensitivity": [EPS, 100.0],
                "lpp_threshold": [-5.0, 5.0],
                "lpp_slope": [EPS, 10.0],
            },
        },
    },
    # Baseline CMR (no stop rule, no emotion/LPP channels)
    {
        "enabled": False,
        "model_name": "WeirdCMRNoStop",
        "make_factory_path": "jaxcmr.models.cmr.make_factory",
        "parameters": {
            "fixed": _NO_STOP_FIXED,
            "free": {**BASE_FREE},
        },
    },
    # Emotion-only CMR (no LPP terms, single-channel encoding)
    {
        "enabled": False,
        "model_name": "EEGEmotionOnly",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
                **_NO_LPP_FIXED,
            },
            "free": {**ECMR_FREE},
        },
    },
    # Emotion-only CMR with multiplicative emotion-primacy rule
    {
        "enabled": False,
        "model_name": "EEGEmotionOnlyMultiplicative",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": True,
                **_NO_LPP_FIXED,
            },
            "free": {**ECMR_FREE},
        },
    },
    # LPP-only CMR (no emotion main effect; linear LPP main, no interaction)
    {
        "enabled": False,
        "model_name": "EEGLPPOnly",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
                "emotion_scale": 0.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
            },
            "free": {
                **BASE_FREE,
                "lpp_main_scale": [EPS, 100.0],
                "lpp_main_threshold": [-5.0, 5.0],
            },
        },
    },
    # LPP-only CMR, parsimonious (threshold fixed to 0; k=10)
    {
        "enabled": True,
        "model_name": "EEGLPPParsimonious",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
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
    # LPP-only CMR with nonlinear (exponent) LPP main + interaction terms
    {
        "enabled": False,
        "model_name": "EEGLPPExponentOnly",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
                "emotion_scale": 0.0,
                "lpp_main_scale": 1.0,
                "lpp_inter_scale": 0.0,
                "lpp_inter_threshold": 0.0,
                "lpp_inter_exponent": 1.0,
            },
            "free": {
                **BASE_FREE,
                "lpp_main_exponent": [0.5, 50.0],
                "lpp_main_threshold": [-5.0, 5.0],
            },
        },
    },
    # EEG-CMR main effects (emotion + LPP main, no interaction)
    {
        "enabled": False,
        "model_name": "EEGMainEffects",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
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
    # EEG-CMR main effects + interaction
    {
        "enabled": False,
        "model_name": "EEGMainEffectsPlusInteraction",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
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
    # EEG-CMR nonlinear main effects + interaction (exponent on LPP)
    {
        "enabled": False,
        "model_name": "EEGEmotionLPPExponentPlusInteraction",
        "make_factory_path": "lpp_ecmr.models.eeg_cmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
                "lpp_main_scale": 1.0,
                "lpp_inter_scale": 1.0,
            },
            "free": {
                **BASE_FREE,
                "emotion_scale": [0.0, 10.0],
                "lpp_main_exponent": [0.5, 50.0],
                "lpp_inter_exponent": [0.5, 50.0],
                "lpp_main_threshold": [-5.0, 5.0],
                "lpp_inter_threshold": [-5.0, 5.0],
            },
        },
    },
    # EEG-eCMR two-layer main effects
    {
        "enabled": False,
        "model_name": "EEGTwoLayerMainEffects",
        "make_factory_path": "lpp_ecmr.models.eeg_ecmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
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
    # EEG-eCMR two-layer main effects + interaction
    {
        "enabled": False,
        "model_name": "EEGTwoLayerInteraction",
        "make_factory_path": "lpp_ecmr.models.eeg_ecmr.make_factory",
        "parameters": {
            "fixed": {
                **_NO_STOP_FIXED,
                "modulate_emotion_by_primacy": False,
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
]
