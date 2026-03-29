"""Shared fitting configuration for lpp_ecmr render notebooks."""

__all__ = [
    "EPS",
    "BASE_FREE",
    "ECMR_FREE",
    "STOP_FREE",
    "BASE_PARAMS",
    "ECMR_BASE_PARAMS",
]


# Machine epsilon - lower bound for continuous parameters
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

# Stop parameters for termination-enabled model variants
STOP_FREE = {
    "stop_probability_scale": [EPS, 1.0],
    "stop_probability_growth": [EPS, 10.0],
}

# Shared fixed parameters for models that disable stopping/context-update
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
    "loss_fn_path": "jaxcmr.loss.set_permutation_likelihood.ExcludeTerminationLikelihoodFnGenerator",
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
            "kwargs": {
                "category_field": "condition",
                "category_values": [1, 2],
                "labels": ["Negative", "Neutral"],
            },
            "ylim": [0.2, 0.8],
            "color_cycle": ["red", "black"],
        },
        {
            "target": "jaxcmr.analyses.cat_lpp_by_recall.plot_cat_lpp_by_recall",
            "figure_suffix": "cat_lpp_by_recall_NEGATIVE_EARLYLPP",
            "kwargs": {
                "category_field": "condition",
                "labels": ["Recalled", "Unrecalled"],
                "category_value": 1,
                "contrast_name": "Negative",
                "lpp_field": "EarlyLPP",
            },
            "ylim": [-0.6, 2.2],
        },
        {
            "target": "jaxcmr.analyses.cat_lpp_by_recall.plot_cat_lpp_by_recall",
            "figure_suffix": "cat_lpp_by_recall_NEUTRAL_EARLYLPP",
            "kwargs": {
                "category_field": "condition",
                "labels": ["Recalled", "Unrecalled"],
                "category_value": 2,
                "contrast_name": "Neutral",
                "lpp_field": "EarlyLPP",
            },
            "ylim": [-0.6, 2.2],
        },
        {
            "target": "jaxcmr.analyses.cat_lpp_by_recall.plot_cat_lpp_by_recall",
            "figure_suffix": "cat_lpp_by_recall",
            "kwargs": {
                "category_field": "condition",
                "labels": [
                    "Recalled Negative",
                    "Unrecalled Negative",
                    "Recalled Neutral",
                    "Unrecalled Neutral",
                ],
                "category_value": [2, 1, 4, 3],
                "contrast_name": "Condition x Recall",
                "lpp_field": "EarlyLPP",
                "exclude_ci": True,
            },
            "ylim": [-0.6, 2.2],
        },
    ],
}

# eCMR-specific base params (adds emotion component paths)
ECMR_BASE_PARAMS = {
    **BASE_PARAMS,
    "base_run_tag": "50_set_likelihood",
    "component_paths": {
        **BASE_PARAMS["component_paths"],
    },
}
