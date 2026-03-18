"""Centralized configuration for lpp_ecmr analyses.

Shared parameter bounds, base fitting parameters, and analysis configs
used by render_* orchestrator notebooks.
"""

# Machine epsilon — lower bound for continuous parameters
EPS = 2.220446049250313e-16

# Standard free parameter bounds shared across CMR-family models
BASE_FREE = {
    "encoding_drift_rate": [EPS, 0.9999999999999998],
    "start_drift_rate": [EPS, 0.9999999999999998],
    "recall_drift_rate": [EPS, 0.9999999999999998],
    "shared_support": [EPS, 100.0],
    "item_support": [EPS, 100.0],
    "learning_rate": [EPS, 0.9999999999999998],
    "primacy_scale": [EPS, 100.0],
    "primacy_decay": [EPS, 100.0],
    "choice_sensitivity": [EPS, 100.0],
}

# eCMR adds emotion_scale to the base set
ECMR_FREE = {**BASE_FREE, "emotion_scale": [EPS, 10.0]}

# Default fitting hyperparameters
BASE_PARAMS = {
    "redo_fits": False,
    "redo_sims": True,
    "redo_figures": True,
    "handle_elis": False,
    "filter_repeated_recalls": True,
    "base_run_tag": "50_set_likelihood_fixed_term",
    "experiment_count": 200,
    "max_subjects": 0,
    "base_data_tag": "TalmiEEG",
    "data_tag": "TalmiEEG",
    "data_path": "data/TalmiEEG.h5",
    "trial_query": "data['subject'] > -1",
    "target_directory": "",
    "component_paths": {
        "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
        "context_create_fn": "jaxcmr.components.context.init",
        "termination_policy_create_fn": "jaxcmr.components.termination.NoStopTermination",
    },
    "sim_alg_path": "jaxcmr.simulation.simulate_study_free_recall_and_forced_stop",
    "loss_fn_path": "jaxcmr.loss.set_permutation_likelihood.MemorySearchLikelihoodFnGenerator",
    "fit_alg_path": "jaxcmr.fitting.ScipyDE",
    "seed": 0,
    "relative_tolerance": 0.001,
    "popsize": 15,
    "num_steps": 1000,
    "cross_rate": 0.9,
    "diff_w": 0.85,
    "best_of": 3,
    "comparison_analysis_configs": [
        {
            "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
            "figure_suffix": "cat_spc_negative",
            "kwargs": {"category_field": "condition", "category_values": [1]},
            "ylim": [0.2, 0.8],
        },
        {
            "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
            "figure_suffix": "cat_spc_neutral",
            "kwargs": {"category_field": "condition", "category_values": [2]},
            "ylim": [0.2, 0.8],
        },
        {"target": "jaxcmr.analyses.spc.plot_spc", "figure_suffix": "spc"},
        {"target": "jaxcmr.analyses.crp.plot_crp", "figure_suffix": "crp"},
        {"target": "jaxcmr.analyses.pnr.plot_pnr", "figure_suffix": "pnr"},
    ],
    "single_analysis_configs": [
        {
            "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
            "figure_suffix": "cat_spc",
            "kwargs": {"category_field": "condition"},
        },
        {
            "target": "jaxcmr.analyses.cat_lpp_spc.plot_cat_lpp_spc",
            "figure_suffix": "cat_lpp_spc",
            "kwargs": {"lpp_field": "EarlyLPP"},
        },
        {
            "target": "jaxcmr.analyses.cat_lpp_by_recall.plot_cat_lpp_by_recall",
            "figure_suffix": "cat_lpp_by_recall",
            "kwargs": {
                "category_field": "condition",
                "category_value": 1,
                "lpp_field": "EarlyLPP",
                "contrast_name": "NEGATIVE_EARLYLPP",
            },
            "ylim": [-0.1, 1.5],
            "color_cycle": ["C2", "C3"],
        },
    ],
}

# eCMR-specific base params (adds emotion component paths)
ECMR_BASE_PARAMS = {
    **BASE_PARAMS,
    "base_run_tag": "50_set_likelihood",
    "component_paths": {
        **BASE_PARAMS["component_paths"],
        "emotion_mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "emotion_mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
    },
}

# Shared fixed parameters for models that disable stopping/context-update
# Cross-validation base params (different hyperparams from fitting)
CV_BASE_PARAMS = {
    "redo_cv": True,
    "max_subjects": 0,
    "data_tag": "TalmiEEG",
    "data_path": "data/TalmiEEG.h5",
    "trial_query": "data['subject'] > -1",
    "target_directory": "",
    "component_paths": {
        "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
        "context_create_fn": "jaxcmr.components.context.init",
        "termination_policy_create_fn": "jaxcmr.components.termination.NoStopTermination",
    },
    "loss_fn_path": "jaxcmr.loss.set_permutation_likelihood.MemorySearchLikelihoodFnGenerator",
    "fit_alg_path": "jaxcmr.fitting.ScipyDE",
    "fold_field": "list",
    "cv_best_of": 1,
    "base_run_tag": "50_set_likelihood_fixed_term",
    "best_of": 3,
    "relative_tolerance": 0.001,
    "popsize": 15,
    "num_steps": 1000,
    "cross_rate": 0.9,
    "diff_w": 0.85,
}

CV_ECMR_BASE_PARAMS = {
    **CV_BASE_PARAMS,
    "base_run_tag": "50_set_likelihood",
    "component_paths": {
        **CV_BASE_PARAMS["component_paths"],
        "emotion_mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
        "emotion_mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
    },
}


def get_models_by_name(model_list, *names):
    """Select model configs by name from a model list."""
    by_name = {m["model_name"]: m for m in model_list}
    return [by_name[n] for n in names]


_NO_STOP_FIXED = {
    "allow_repeated_recalls": False,
    "learn_after_context_update": False,
}

_NO_LPP_FIXED = {
    "lpp_main_scale": 0.0,
    "lpp_main_threshold": 0.0,
    "lpp_inter_scale": 0.0,
    "lpp_inter_threshold": 0.0,
}

# ---------------------------------------------------------------------------
# Single-context models (render_model_fitting.ipynb)
# ---------------------------------------------------------------------------

SINGLE_CONTEXT_MODELS = [
    # Strength-only EEG model (no context dynamics)
    {
        "enabled": False,
        "model_name": "EEGStrength",
        "make_factory_path": "jaxcmr.models_eeg.eeg_strength.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_strength.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_cmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_ecmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_ecmr.make_factory",
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

# ---------------------------------------------------------------------------
# Full eCMR models (render_model_fitting_ecmr.ipynb)
# ---------------------------------------------------------------------------

ECMR_MODELS = [
    # eCMR Emotion-only (no LPP terms)
    {
        "enabled": False,
        "model_name": "eCMREmotionOnly",
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
    # eCMR Emo×LPP — parsimonious: 1 LPP param, emotional items only, no threshold
    {
        "enabled": False,
        "model_name": "eCMREmotionTimesLPP",
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
    # eCMR Emo×LPP Broad — phi_emot modulates both temporal and emotional MCF
    {
        "enabled": True,
        "model_name": "eCMREmotionTimesLPPBroad",
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
    # eCMR Emotion Broad — phi_emot on both pathways, no LPP (matched baseline)
    {
        "enabled": True,
        "model_name": "eCMREmotionBroad",
        "make_factory_path": "jaxcmr.models_eeg.eeg_full_ecmr.make_factory",
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
