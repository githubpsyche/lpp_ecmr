# Pooled model figures

This package contains the two matched empirical-versus-predicted figures used
by `index.qmd`. Both figures compare the same four sources:

1. observed data;
2. an LPP-based learning boost for negative items only;
3. an emotion-based learning boost; and
4. both boosts together.

The shared model set makes the complementary roles visible. Emotion-based
learning produces the negative-item recall advantage, while negative-item LPP
modulation produces the remembered-minus-forgotten Early-LPP separation.

## Inputs

The recall-rate figure reads the verified
`../pooled_model_summaries/diagnostic_summary.csv` and
`../pooled_model_summaries/diagnostic_contrasts.csv` products.

The Early-LPP figure reads:

- `original_early_lpp_summary.csv`;
- `original_early_lpp_contrasts.csv`; and
- the source record `original_early_lpp_source.json`.

`build_original_early_lpp_summaries.py` reproduces the original-z empirical
summaries from Robin Hellerstedt's mixed-list analysis delivery and summarizes
the six relevant mixed-list simulations. It authenticates the two source
members against the hashes recorded in `original_early_lpp_source.json`.

The builder first looks for:

`/Users/jordangunn/workspace/downloads/LPP_Zarubin_PureLists_for_Deborah.zip`

If that redundant ZIP is absent, it uses:

`/Users/jordangunn/workspace/downloads/LPP_Zarubin_PureLists_for_Deborah_extracted_2026-07-17`

Set `LPP_ECMR_EMPIRICAL_SOURCE_ROOT` to use a different extracted delivery.
The builder writes the participant-level audit table
`empirical_early_lpp_participant_means.csv` alongside the retained summaries.

## Outputs

- `matched_models_recall_summary.csv`
- `matched_models_recall_rate_figure.{png,svg}`
- `matched_models_early_lpp_summary.csv`
- `matched_models_early_lpp_figure.{png,svg}`

Observed intervals are participant-bootstrap 95% intervals. Predicted
intervals are central 95% intervals across 200 complete simulated datasets.

## Rebuild

From the project root:

```bash
python work/pooled_model_summaries/build_results.py --no-prose
python work/pooled_model_figures/build_original_early_lpp_summaries.py
python work/pooled_model_figures/build_matched_models_recall_figure.py
python work/pooled_model_figures/build_matched_models_early_lpp_figure.py
```

These builders read the fitted and simulated artifacts but do not refit models
or regenerate simulations.
