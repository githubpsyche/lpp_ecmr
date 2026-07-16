"""Validate the boundary-search candidate matching the qualitative contrasts."""

from __future__ import annotations

import csv
import json

import tune_full_for_mixed_pure_lpp as base
from jaxcmr.helpers import load_data


INPUT = base.WORK_DIR / f"{base.MODEL}_recall_competition_boundary_candidates.csv"
OUTPUT = base.WORK_DIR / (
    f"{base.MODEL}_recall_competition_qualitative_candidate.json"
)


def main() -> None:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if row["stage"] == "boundary_refine"
        and float(row["mixed_negative_delta"]) > float(row["mixed_neutral_delta"])
        and 0 <= float(row["mixed_neutral_delta"]) <= 0.05
        and float(row["pure_negative_delta"]) > 0
        and float(row["pure_neutral_delta"]) >= float(row["pure_negative_delta"])
        and float(row["pure_neutral_delta"]) >= 0.08
    ]
    if not eligible:
        raise RuntimeError("No refined candidate meets the qualitative criteria")
    selected = max(eligible, key=lambda row: float(row["mixed_negative_delta"]))
    candidate = base.candidate_from_row(selected)

    source, fixed = base.load_source_fit()
    mixed_data = load_data(base.DATA_PATH)
    pure_data = base.make_pure_data(mixed_data)
    evaluator = base.Evaluator(mixed_data, pure_data, fixed)
    result = evaluator.evaluate(candidate, experiment_count=200, seed=9450)

    output = {
        "status": "simulation-tuned mechanism-capability configuration; not a likelihood fit",
        "selection_rule": (
            "From 20-repetition boundary-refine results: mixed negative delta > "
            "mixed neutral delta; mixed neutral delta in [0, .05]; both pure "
            "deltas positive; pure neutral delta >= pure negative delta; pure "
            "neutral delta >= .08. Select largest mixed negative delta."
        ),
        "selected_refine_metrics": {
            key: float(selected[key])
            for key in (
                "mixed_negative_delta",
                "mixed_neutral_delta",
                "pure_negative_delta",
                "pure_neutral_delta",
            )
        },
        "parameters": candidate,
        "validation_experiment_count": 200,
        "validation_seed": 9450,
        "validation_metrics": result,
        "pure_input_constraint": base.pure_input_metadata(mixed_data, pure_data),
    }
    with OUTPUT.open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
