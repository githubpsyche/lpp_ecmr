"""Search Full-model parameters for mixed-versus-pure LPP selection effects.

This is a mechanism-capability search, not a likelihood refit. All 14 free
parameters of EEM_eCMR_LPP_Full are varied. Candidate configurations are
ranked by whether recall competition produces:

* a substantial negative-item LPP difference but little neutral-item LPP
  difference in mixed lists; and
* positive negative- and neutral-item LPP differences in pure lists, with the
  neutral difference at least as large as the negative difference.

The strict category-level ordering of the four LPP bins is recorded but is not
optimized. The synthetic pure-list inputs have nearly identical fixed negative
and neutral LPP means, so that ordering cannot coexist with substantial
positive recall differences in both categories (see the output metadata).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random
from jaxcmr.components.context import init as init_context
from jaxcmr.components.linear_memory import init_mcf, init_mfc
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import generate_trial_mask, load_data
from jaxcmr.simulation import (
    simulate_h5_from_h5,
    simulate_study_free_recall_and_forced_stop,
)
from scipy.stats import qmc

from lpp_ecmr.model_comparison_registry import LPP_SLOPE_BOUNDS
from lpp_ecmr.models.full_eeg_ecmr import make_factory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
MODEL = "EEM_eCMR_LPP_Full"
SOURCE_FIT = (
    PROJECT_ROOT / "work" / "lpp_model_comparison" / f"{MODEL}_best_of_3.json"
)
DATA_PATH = PROJECT_ROOT / "data" / "TalmiEEG.h5"
PURE_DESIGN_PATH = WORK_DIR / "pure_list_design.csv"
OUTPUT_STEM = WORK_DIR / f"{MODEL}_recall_competition_search"

FREE_PARAMETERS = (
    "encoding_drift_rate",
    "start_drift_rate",
    "recall_drift_rate",
    "shared_support",
    "item_support",
    "learning_rate",
    "primacy_scale",
    "primacy_decay",
    "choice_sensitivity",
    "emotion_scale",
    "lpp_main_scale",
    "lpp_inter_scale",
    "source_encoding_drift_rate",
    "source_learning_rate",
)
UNIT_PARAMETERS = {
    "encoding_drift_rate",
    "start_drift_rate",
    "recall_drift_rate",
    "learning_rate",
    "source_encoding_drift_rate",
}
LOG_RANGES = {
    "shared_support": (0.1, 100.0),
    "item_support": (0.1, 100.0),
    "primacy_scale": (0.05, 20.0),
    "primacy_decay": (0.05, 20.0),
    "choice_sensitivity": (0.1, 20.0),
    "emotion_scale": (1.0, 10.0),
    "source_learning_rate": (0.1, 10.0),
}

BROAD_CANDIDATES = 144
SEARCH_EXPERIMENTS = 3
REFINE_COUNT = 12
REFINE_EXPERIMENTS = 20
FINALIST_COUNT = 4
FINALIST_EXPERIMENTS = 80
VALIDATION_EXPERIMENTS = 200


def load_source_fit() -> tuple[dict[str, float], dict[str, object]]:
    """Load the pooled Full fit as the search center and fixed configuration."""
    with SOURCE_FIT.open() as handle:
        result = json.load(handle)
    values = {key: float(entries[0]) for key, entries in result["fits"].items()}
    fixed = {
        key: value
        for key, value in values.items()
        if key not in FREE_PARAMETERS and key != "subject"
    }
    return values, fixed


def make_pure_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the established equal-count synthetic pure-list inputs."""
    with PURE_DESIGN_PATH.open(newline="") as handle:
        design = list(csv.DictReader(handle))
    conditions = np.asarray(
        [int(row["condition"]) for row in design], dtype=np.int32
    )
    trial_indices = np.asarray(
        [int(row["trial_index"]) for row in design], dtype=np.int32
    )
    if not np.array_equal(trial_indices, np.arange(len(conditions))):
        raise ValueError("pure_list_design.csv does not match the data trial order")

    fields = (
        "EarlyLPP",
        "list",
        "listLength",
        "pres_itemids",
        "pres_itemnos",
        "recalls",
        "subject",
    )
    pure = {key: np.array(data[key], copy=True) for key in fields}
    pure["condition"] = np.repeat(
        conditions[:, np.newaxis], np.asarray(data["condition"]).shape[1], axis=1
    ).astype(np.int32)
    pure["list_type"] = conditions[:, np.newaxis]
    return pure


def sample_candidates(source: dict[str, float]) -> list[dict[str, float]]:
    """Combine broad Latin-hypercube draws with local perturbations."""
    broad_count = 96
    local_count = BROAD_CANDIDATES - broad_count - 1
    sampler = qmc.LatinHypercube(d=len(FREE_PARAMETERS), seed=3107)
    draws = sampler.random(broad_count)
    candidates: list[dict[str, float]] = [
        {key: source[key] for key in FREE_PARAMETERS}
    ]

    for draw in draws:
        candidate: dict[str, float] = {}
        for name, value in zip(FREE_PARAMETERS, draw, strict=True):
            if name in UNIT_PARAMETERS:
                candidate[name] = 0.01 + 0.98 * float(value)
            elif name in LOG_RANGES:
                low, high = LOG_RANGES[name]
                candidate[name] = float(
                    math.exp(math.log(low) + value * math.log(high / low))
                )
            elif name in {"lpp_main_scale", "lpp_inter_scale"}:
                candidate[name] = float(
                    LPP_SLOPE_BOUNDS[0]
                    + value * (LPP_SLOPE_BOUNDS[1] - LPP_SLOPE_BOUNDS[0])
                )
            else:
                raise KeyError(name)
        candidates.append(candidate)

    rng = np.random.default_rng(811)
    for _ in range(local_count):
        candidate = {}
        for name in FREE_PARAMETERS:
            center = source[name]
            if name in UNIT_PARAMETERS:
                clipped = np.clip(center, 1e-4, 1 - 1e-4)
                logit = math.log(clipped / (1 - clipped))
                candidate[name] = float(1 / (1 + math.exp(-(logit + rng.normal(0, 1.2)))))
            elif name in LOG_RANGES:
                low, high = LOG_RANGES[name]
                candidate[name] = float(
                    np.clip(center * math.exp(rng.normal(0, 1.0)), low, high)
                )
            elif name in {"lpp_main_scale", "lpp_inter_scale"}:
                # Use the complete slope range locally because the fitted main
                # effect is near zero and pure-neutral selection requires it.
                candidate[name] = float(rng.uniform(*LPP_SLOPE_BOUNDS))
            else:
                raise KeyError(name)
        candidates.append(candidate)
    return candidates


def recalled_items(recalls: np.ndarray) -> np.ndarray:
    recalled = np.zeros_like(recalls, dtype=bool)
    rows = np.repeat(np.arange(recalls.shape[0]), recalls.shape[1])
    positions = recalls.reshape(-1).astype(int)
    valid = positions > 0
    recalled[rows[valid], positions[valid] - 1] = True
    return recalled


def cell_means(sim: dict[str, np.ndarray]) -> dict[str, float]:
    """Return participant-averaged LPP means for the four recall cells."""
    lpp = np.asarray(sim["EarlyLPP"])
    condition = np.asarray(sim["condition"])
    recalled = recalled_items(np.asarray(sim["recalls"]))
    subjects = np.asarray(sim["subject"]).reshape(-1)
    out: dict[str, float] = {}
    for category, category_name in ((1, "negative"), (2, "neutral")):
        category_recall_rates = []
        for subject in np.unique(subjects):
            category_mask = (subjects[:, np.newaxis] == subject) & (
                condition == category
            )
            category_recall_rates.append(
                float(np.count_nonzero(recalled & category_mask) / np.count_nonzero(category_mask))
            )
        out[f"{category_name}_recall_rate"] = float(
            np.mean(category_recall_rates)
        )
        for is_recalled, recall_name in ((True, "recalled"), (False, "forgotten")):
            participant_means = []
            for subject in np.unique(subjects):
                mask = (
                    (subjects[:, np.newaxis] == subject)
                    & (condition == category)
                    & (recalled == is_recalled)
                )
                if not np.any(mask):
                    return {
                        "negative_recalled": math.nan,
                        "negative_forgotten": math.nan,
                        "neutral_recalled": math.nan,
                        "neutral_forgotten": math.nan,
                        "negative_recall_rate": math.nan,
                        "neutral_recall_rate": math.nan,
                    }
                participant_means.append(float(np.mean(lpp[mask])))
            out[f"{category_name}_{recall_name}"] = float(
                np.mean(participant_means)
            )
    return out


def metrics(mixed: dict[str, float], pure: dict[str, float]) -> dict[str, float]:
    """Calculate the requested recall contrasts and category-ordering margins."""
    out = {
        "mixed_negative_delta": mixed["negative_recalled"]
        - mixed["negative_forgotten"],
        "mixed_neutral_delta": mixed["neutral_recalled"]
        - mixed["neutral_forgotten"],
        "pure_negative_delta": pure["negative_recalled"]
        - pure["negative_forgotten"],
        "pure_neutral_delta": pure["neutral_recalled"]
        - pure["neutral_forgotten"],
        "mixed_cross_category_margin": min(
            mixed["negative_recalled"], mixed["negative_forgotten"]
        )
        - max(mixed["neutral_recalled"], mixed["neutral_forgotten"]),
        "pure_cross_category_margin": min(
            pure["negative_recalled"], pure["negative_forgotten"]
        )
        - max(pure["neutral_recalled"], pure["neutral_forgotten"]),
    }
    out |= {f"mixed_{key}": value for key, value in mixed.items()}
    out |= {f"pure_{key}": value for key, value in pure.items()}
    return out


def capability_score(values: dict[str, float]) -> float:
    """Score the composition-dependent recall-selection requirements."""
    if not all(np.isfinite(value) for value in values.values()):
        return 1e6
    dmn = values["mixed_negative_delta"]
    dmu = values["mixed_neutral_delta"]
    dpn = values["pure_negative_delta"]
    dpu = values["pure_neutral_delta"]

    violations = (
        max(0.0, 0.25 - dmn) / 0.25,
        max(0.0, abs(dmu) - 0.05) / 0.10,
        max(0.0, -dmu) / 0.05,
        max(0.0, 0.20 - dpn) / 0.20,
        max(0.0, 0.20 - dpu) / 0.20,
        max(0.0, dpn - dpu) / 0.10,
        max(0.0, 0.15 - (dpu - dmu)) / 0.15,
        max(0.0, -values["mixed_cross_category_margin"]) / 0.20,
    )
    return float(sum(value * value for value in violations))


class Evaluator:
    """Evaluate candidates under common random numbers at a chosen replication count."""

    def __init__(
        self,
        mixed_data: dict[str, np.ndarray],
        pure_data: dict[str, np.ndarray],
        fixed: dict[str, object],
    ) -> None:
        self.mixed_data = mixed_data
        self.pure_data = pure_data
        self.fixed = fixed
        self.factory = make_factory(
            mfc_create_fn=init_mfc,
            mcf_create_fn=init_mcf,
            context_create_fn=init_context,
            termination_policy_create_fn=NoStopTermination,
        )
        self.unique_subjects = jnp.unique(jnp.asarray(mixed_data["subject"]))

    def _parameters(self, candidate: dict[str, float]) -> dict[str, jnp.ndarray]:
        pooled = self.fixed | candidate
        return {
            key: jnp.repeat(jnp.asarray([value]), self.unique_subjects.shape[0])
            for key, value in pooled.items()
        } | {"subject": self.unique_subjects}

    def _simulate(
        self,
        data: dict[str, np.ndarray],
        parameters: dict[str, jnp.ndarray],
        experiment_count: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        return simulate_h5_from_h5(
            self.factory,
            data,
            None,
            parameters,
            generate_trial_mask(data, "data['subject'] > -1"),
            experiment_count,
            random.PRNGKey(seed),
            simulate_trial_fn=simulate_study_free_recall_and_forced_stop,
        )

    def evaluate(
        self,
        candidate: dict[str, float],
        experiment_count: int,
        seed: int,
    ) -> dict[str, float]:
        parameters = self._parameters(candidate)
        mixed = cell_means(
            self._simulate(self.mixed_data, parameters, experiment_count, seed)
        )
        pure = cell_means(
            self._simulate(self.pure_data, parameters, experiment_count, seed + 1)
        )
        values = metrics(mixed, pure)
        values["score"] = capability_score(values)
        return values


def evaluate_stage(
    evaluator: Evaluator,
    candidates: list[dict[str, float]],
    experiment_count: int,
    seed: int,
    stage: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    start = time.time()
    for index, candidate in enumerate(candidates):
        result = evaluator.evaluate(candidate, experiment_count, seed)
        row: dict[str, float | int | str] = {
            "stage": stage,
            "candidate": index,
            "experiment_count": experiment_count,
            **candidate,
            **result,
        }
        rows.append(row)
        if index % 12 == 0 or index == len(candidates) - 1:
            print(
                f"{stage}: {index + 1}/{len(candidates)}; "
                f"best score={min(float(r['score']) for r in rows):.4f}; "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return rows


def candidate_from_row(row: dict[str, float | int | str]) -> dict[str, float]:
    return {name: float(row[name]) for name in FREE_PARAMETERS}


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pure_input_metadata(
    data: dict[str, np.ndarray], pure_data: dict[str, np.ndarray]
) -> dict[str, object]:
    conditions = np.asarray(pure_data["condition"])[:, 0]
    lpp = np.asarray(data["EarlyLPP"])
    recalls = np.count_nonzero(np.asarray(data["recalls"]), axis=1)
    means = {str(category): float(lpp[conditions == category].mean()) for category in (1, 2)}
    recall_rates = {
        str(category): float(
            recalls[conditions == category].sum()
            / (lpp.shape[1] * np.count_nonzero(conditions == category))
        )
        for category in (1, 2)
    }
    return {
        "fixed_lpp_means": means,
        "negative_minus_neutral_lpp_mean": means["1"] - means["2"],
        "fixed_recall_rates": recall_rates,
        "ordering_bound": (
            "Because both pure-list categories have the same fixed recall rate, "
            "strict separation of both negative bins above both neutral bins "
            "implies r*negative_delta + (1-r)*neutral_delta cannot exceed "
            "the fixed category-mean difference."
        ),
    }


def main() -> None:
    source, fixed = load_source_fit()
    mixed_data = load_data(DATA_PATH)
    pure_data = make_pure_data(mixed_data)
    evaluator = Evaluator(mixed_data, pure_data, fixed)

    candidates = sample_candidates(source)
    search_rows = evaluate_stage(
        evaluator,
        candidates,
        SEARCH_EXPERIMENTS,
        seed=4310,
        stage="search",
    )
    search_ranked = sorted(search_rows, key=lambda row: float(row["score"]))

    refine_candidates = [
        candidate_from_row(row) for row in search_ranked[:REFINE_COUNT]
    ]
    refine_rows = evaluate_stage(
        evaluator,
        refine_candidates,
        REFINE_EXPERIMENTS,
        seed=5320,
        stage="refine",
    )
    refine_ranked = sorted(refine_rows, key=lambda row: float(row["score"]))

    finalist_candidates = [
        candidate_from_row(row) for row in refine_ranked[:FINALIST_COUNT]
    ]
    finalist_rows = evaluate_stage(
        evaluator,
        finalist_candidates,
        FINALIST_EXPERIMENTS,
        seed=6330,
        stage="finalist",
    )
    finalist_ranked = sorted(finalist_rows, key=lambda row: float(row["score"]))

    best_candidate = candidate_from_row(finalist_ranked[0])
    validation_rows = evaluate_stage(
        evaluator,
        [best_candidate],
        VALIDATION_EXPERIMENTS,
        seed=7340,
        stage="validation",
    )
    best_row = validation_rows[0]

    all_rows = search_rows + refine_rows + finalist_rows + validation_rows
    write_csv(all_rows, OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_candidates.csv"))

    config = {
        "status": "simulation-tuned mechanism-capability configuration; not a likelihood fit",
        "model": MODEL,
        "source_fit": str(SOURCE_FIT.relative_to(PROJECT_ROOT)),
        "search_scope": list(FREE_PARAMETERS),
        "fixed_parameters": fixed,
        "search_design": {
            "broad_candidates": BROAD_CANDIDATES,
            "search_experiments": SEARCH_EXPERIMENTS,
            "refine_count": REFINE_COUNT,
            "refine_experiments": REFINE_EXPERIMENTS,
            "finalist_count": FINALIST_COUNT,
            "finalist_experiments": FINALIST_EXPERIMENTS,
            "validation_experiments": VALIDATION_EXPERIMENTS,
        },
        "pure_input_constraint": pure_input_metadata(mixed_data, pure_data),
        "best_parameters": best_candidate,
        "validation_metrics": {
            key: value
            for key, value in best_row.items()
            if key not in FREE_PARAMETERS
        },
    }
    with OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_best_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    print(json.dumps(config["validation_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
