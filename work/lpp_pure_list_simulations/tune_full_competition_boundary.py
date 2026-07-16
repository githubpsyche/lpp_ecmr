"""Test whether recall competition alone can create list-composition gating.

This supplements ``tune_full_for_mixed_pure_lpp.py`` by fixing the Full
model's emotional LPP increment to its allowed zero boundary while varying all
13 remaining free parameters. At this boundary negative and neutral items have
the same direct LPP slope, so any mixed-list emotion specificity must emerge
from the memory dynamics rather than from an explicit emotion-by-LPP term.
"""

from __future__ import annotations

import json
from pathlib import Path

import tune_full_for_mixed_pure_lpp as base
from jaxcmr.helpers import load_data


OUTPUT_STEM = base.WORK_DIR / f"{base.MODEL}_recall_competition_boundary"
SEARCH_COUNT = 108
REFINE_COUNT = 8
FINALIST_COUNT = 2


def main() -> None:
    source, fixed = base.load_source_fit()
    mixed_data = load_data(base.DATA_PATH)
    pure_data = base.make_pure_data(mixed_data)
    evaluator = base.Evaluator(mixed_data, pure_data, fixed)

    candidates = base.sample_candidates(source)[:SEARCH_COUNT]
    for candidate in candidates:
        candidate["lpp_inter_scale"] = 0.0

    search_rows = base.evaluate_stage(
        evaluator,
        candidates,
        experiment_count=3,
        seed=8410,
        stage="boundary_search",
    )
    search_ranked = sorted(search_rows, key=lambda row: float(row["score"]))

    refine_candidates = [
        base.candidate_from_row(row) for row in search_ranked[:REFINE_COUNT]
    ]
    refine_rows = base.evaluate_stage(
        evaluator,
        refine_candidates,
        experiment_count=20,
        seed=8420,
        stage="boundary_refine",
    )
    refine_ranked = sorted(refine_rows, key=lambda row: float(row["score"]))

    finalist_candidates = [
        base.candidate_from_row(row) for row in refine_ranked[:FINALIST_COUNT]
    ]
    finalist_rows = base.evaluate_stage(
        evaluator,
        finalist_candidates,
        experiment_count=80,
        seed=8430,
        stage="boundary_finalist",
    )
    finalist_ranked = sorted(finalist_rows, key=lambda row: float(row["score"]))

    best_candidate = base.candidate_from_row(finalist_ranked[0])
    validation_rows = base.evaluate_stage(
        evaluator,
        [best_candidate],
        experiment_count=200,
        seed=8440,
        stage="boundary_validation",
    )
    best_row = validation_rows[0]

    all_rows = search_rows + refine_rows + finalist_rows + validation_rows
    base.write_csv(
        all_rows, OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_candidates.csv")
    )
    config = {
        "status": "simulation-tuned mechanism-capability configuration; not a likelihood fit",
        "model": base.MODEL,
        "test": (
            "lpp_inter_scale fixed to zero; mixed-list emotion specificity must "
            "therefore emerge from recall competition"
        ),
        "source_fit": str(base.SOURCE_FIT.relative_to(base.PROJECT_ROOT)),
        "varied_parameters": [
            name for name in base.FREE_PARAMETERS if name != "lpp_inter_scale"
        ],
        "boundary_parameter": {"lpp_inter_scale": 0.0},
        "fixed_parameters": fixed,
        "pure_input_constraint": base.pure_input_metadata(mixed_data, pure_data),
        "best_parameters": best_candidate,
        "validation_metrics": {
            key: value
            for key, value in best_row.items()
            if key not in base.FREE_PARAMETERS
        },
    }
    with OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_best_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(config["validation_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
