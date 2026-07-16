# eCMR model prediction grids

This work package began as a prediction-only comparison of the six pooled
source-context models selected for the manuscript. It now also contains two
smaller empirical-versus-predicted figures chosen to isolate the manuscript's
main mechanistic contrasts. It does not modify the returned fits or simulation
files.

The figures organize the eCMR hierarchy as a two-by-three grid. Rows indicate
whether the **Emotion effect** is absent or present. Columns indicate whether
there is no LPP effect, a **General LPP effect**, or an **Emotion-specific LPP
effect**. These labels are provisional shared vocabulary for the model equation
and diagnostic figures:

- the Emotion effect is the category-wide negative-item term;
- the General LPP effect applies the Early-LPP relationship to negative and
  neutral items; and
- the Emotion-specific LPP effect applies the Early-LPP relationship only to
  negative items.

## Inputs and uncertainty

[`diagnostic_summary.csv`](../lpp_model_results/diagnostic_summary.csv) supplies
the plotted means and central 95% intervals across 200 complete simulated
datasets. The script filters the 12 recall-rate rows and 24 Early-LPP rows for
the six approved models and writes the exact plotted subset to
`prediction_summary.csv`. It checks displayed deltas against
[`diagnostic_contrasts.csv`](../lpp_model_results/diagnostic_contrasts.csv).

No HDF5 simulation is read and no interval is recomputed in this package.

## Outputs

- `recall_rate_prediction_grid.{png,svg}` shows negative and neutral recall-rate
  predictions with negative-minus-neutral delta brackets.
- `early_lpp_prediction_grid.{png,svg}` shows negative remembered, negative
  forgotten, neutral remembered, and neutral forgotten Early-LPP predictions.
  Each category has a remembered-minus-forgotten delta bracket.

Both figures use shared axes, vertical bars, capped intervals, and the coral,
pink, dark-gray, and light-gray palette established in
`work/lpp_pure_list_simulations/plot_mixed_pure_lpp_bars.py`.

Regenerate the figures from the project root with:

```bash
python work/lpp_model_prediction_grids/build_prediction_grids.py
```

The original six-model grids remain exploratory. A combined manuscript figure
and manuscript integration are deferred until the selected contrasts have been
evaluated.

## Selected recall-rate contrast

`selected_recall_rate_figure.{png,svg}` is the first empirical-versus-predicted
revision. It plots observed negative and neutral recall rates beside two models
that both apply an LPP-based learning boost to negative items only. The models
differ only in whether they also include an emotion-based learning boost. This
controlled comparison isolates whether the emotion-based boost recovers the
observed negative-minus-neutral recall-rate difference.

Bars show category means rather than difference scores. Brackets report the
negative-minus-neutral difference. Observed error bars are participant-cluster
bootstrap 95% confidence intervals; predicted error bars are central 95%
intervals across 200 complete simulated datasets. The exact six plotted rows
are written to `selected_recall_summary.csv`.

Regenerate this figure from the project root with:

```bash
python work/lpp_model_prediction_grids/build_selected_recall_figure.py
```

## Selected Early-LPP contrast

`selected_early_lpp_figure.{png,svg}` compares the empirical Early-LPP means
with three predictions that all include an emotion-based learning boost but
vary the LPP-based mechanism: none, a boost fitted for all items, or a boost
fitted for negative items only. Bars show the four category-by-memory-status
means and brackets show remembered-minus-forgotten differences.

Unlike the original prediction grid, this figure reports the original
z-transformed Early-LPP values used in the empirical analysis. The observed
panel is reproduced directly from Robin Hellerstedt's mixed-list analysis code
and data in:

`/Users/jordangunn/workspace/downloads/LPP_Zarubin_PureLists_for_Deborah.zip`

The source members are
`R_Script/Behaviour_and_Relationship_to_LPP.Rmd` and
`Data/Extracted_Behavioural_and_LPP_Data/Single_Trial_Behavioural_and_EEG_Data_Z.csv`.
As in the R code, single-trial values are first averaged within each participant
and condition, then across the 38 participants. Observed intervals use 10,000
participant bootstrap samples. Predicted means use the uncentered `EarlyLPP`
input stored in the returned simulations, summarized first within participant
and then across participants for each of 200 complete simulated datasets.
Predicted intervals are the central 95% range across those datasets.

The empirical source contains 39 rows with `PresentationOrder = 99`. They are
retained in the observed panel, matching the source R code. Because those rows
cannot be assigned a simulated recall outcome, prediction summaries use the
6,469 model positions with observed rather than imputed LPP values. The source
checksums and this qualification are recorded in
`original_early_lpp_source.json`. Derived participant means, all seven sources'
cell summaries, and remembered-minus-forgotten contrasts are stored in
`empirical_early_lpp_participant_means.csv`,
`original_early_lpp_summary.csv`, and
`original_early_lpp_contrasts.csv`, respectively.

Regenerate the figure and summaries from the project root with:

```bash
python work/lpp_model_prediction_grids/build_selected_early_lpp_figure.py
```

### Row-level y-label candidate

`selected_early_lpp_row_ylabels.{png,svg}` preserves the selected four-panel bar
figure but replaces its single figure-level y-axis title with one
`Early LPP amplitude (z)` label on each panel row. The original
`selected_early_lpp_figure.{png,svg}` remains unchanged as the shared-label
candidate.

Regenerate the row-labelled candidate after building the original-scale summary
with:

```bash
python work/lpp_model_prediction_grids/build_selected_early_lpp_row_ylabels.py
```

### Matched-model candidates

`matched_models_recall_rate_figure.{png,svg}` and
`matched_models_early_lpp_figure.{png,svg}` use the same four panels in the
same order:

1. observed data;
2. LPP-based learning boost for negative items only;
3. emotion-based learning boost; and
4. emotion-based plus LPP-based learning boosts, with the LPP-based boost
   applied to negative items only.

The shared model set makes the complementary roles visible across figures. In
the recall-rate figure, the LPP-only specification misses the observed
negative-item recall advantage whereas the emotion-only and combined
specifications reproduce it. In the Early-LPP figure, the emotion-only
specification misses the negative-item remembered-minus-forgotten separation
whereas the LPP-only and combined specifications reproduce it. The all-item
LPP specification remains in the numerical model-comparison and diagnostic
tables rather than either matched-model figure.

Both candidates preserve the existing bar, interval, bracket, palette, and
row-level y-label conventions. Their exact plotted subsets are written to
`matched_models_recall_summary.csv` and
`matched_models_early_lpp_summary.csv`.

Regenerate them with:

```bash
python work/lpp_model_prediction_grids/build_matched_models_recall_figure.py
python work/lpp_model_prediction_grids/build_matched_models_early_lpp_figure.py
```

### Paired-point candidate

`selected_early_lpp_paired_points.{png,svg}` presents the same four panels and
the same means and intervals as the selected bar figure. Within each category,
it connects the forgotten and remembered means so that the line's horizontal
extent directly represents the remembered-minus-forgotten difference. Circles
identify remembered means, squares identify forgotten means, and each row is
labelled with its exact delta to three decimal places. This candidate retains a
shared original-z Early-LPP axis and does not add higher-level comparison
headings.

Regenerate it after building the original-scale summary with:

```bash
python work/lpp_model_prediction_grids/build_selected_early_lpp_paired_points.py
```

### Difference-bar candidate

`selected_early_lpp_difference_bars.{png,svg}` reduces each category to its
remembered-minus-forgotten Early-LPP contrast. Each panel therefore contains
one Negative-item bar and one Neutral-item bar on a shared difference scale.
Error bars are calculated from paired contrasts: participant-bootstrap
contrasts for the observed panel and within-dataset contrasts across the 200
complete simulations for each prediction. They are not reconstructed from the
separate remembered and forgotten intervals. Exact point estimates are printed
to three decimal places.

Regenerate this candidate after building the original-scale summaries with:

```bash
python work/lpp_model_prediction_grids/build_selected_early_lpp_difference_bars.py
```

### Observed-target overlay candidate

`selected_early_lpp_target_overlay.{png,svg}` removes the separate observed
panel and repeats the empirical benchmark inside each of the three prediction
panels. The filled bars and error bars show model predictions and their central
95% simulation intervals. A black horizontal target marks the observed
contrast; the pale surrounding band is its participant-bootstrap 95% confidence
interval. This makes model-versus-observed discrepancies visible within each
panel while retaining separate Negative and Neutral contrasts.

Regenerate this candidate with:

```bash
python work/lpp_model_prediction_grids/build_selected_early_lpp_target_overlay.py
```
