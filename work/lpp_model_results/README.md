# Six-model pooled result package

This flat work unit implements the local artifact package requested by Issue
#10 and the separate exploratory Full-LPP disclosure tracked in Issue #22. It
reads the returned pooled fit and simulation products from
`work/lpp_model_comparison/`; it does not copy, modify, or replace those source
artifacts.

The manuscript-facing primary comparison is restricted to the six approved
source-context models crossing the emotion-based learning boost (absent/present)
with the LPP-based learning boost (none/all items/negative items only).
Source-context Full-LPP fits are retained in a separate exploratory
supplementary table because the manuscript presents the full learning-strength
equation, but they do not enter the primary comparison or figures. Temporal-only
models, subject-wise process-model fits, and a wider-bound sensitivity analysis
remain outside this package.

## Build

From the repository root:

```bash
python3 work/lpp_model_prediction_grids/build_selected_early_lpp_figure.py
python3 work/lpp_model_results/build_results.py
python3 work/lpp_model_results/build_full_lpp_exploratory_table.py
```

The builder checks the selected fits, declared parameter bounds, HDF5 shapes,
simulation storage order, list count, participant count, simulation count, and
within-list LPP centering before producing any summaries. Figure generation
requires R with `ggplot2` and `scales`; `pdftocairo` converts the single ggplot
PDF to SVG without altering its layout.

The original combined diagnostic figure remains reproducible through the main
builder. To generate two alternative, manuscript-sized diagnostic figures
without replacing it, run:

```bash
Rscript work/lpp_model_results/build_split_diagnostic_figures.R
```

## Products

- `model_comparison.csv` and `model_comparison_table.qmd`: six-model NLL, AIC,
  and delta-AIC results.
- `diagnostic_effects_table.qmd`: observed and predicted negative-minus-neutral
  recall differences and remembered-minus-forgotten Early-LPP differences for
  all six models, with each prediction's signed discrepancy from the observed
  contrast. Its Early-LPP columns use the original empirical z-LPP scale from
  `work/lpp_model_prediction_grids/original_early_lpp_contrasts.csv`.
- `parameter_estimates.csv` and `parameter_table.qmd`: complete pooled
  parameter estimates for all six models, including a manuscript-facing model
  key, parameter roles, fitted ranges, fixed values, and the implementation-to-
  manuscript mapping. The table includes the source-recall drift fixed at 1;
  the CSV retains full numerical precision and implementation names.
- `full_lpp_exploratory_table.qmd`: separate supplementary Full-LPP report. Its
  first panel reports the two original Full optimizer results and their nesting
  violations; its second panel preserves the two original Full parameter
  vectors. The already-reported nested-model vectors are not repeated.
- `full_lpp_exploratory_fits.csv` and
  `full_lpp_exploratory_parameters.csv`: complete-precision sources for the two
  panels of the exploratory Full-LPP table.
- `full_lpp_embedded_parent_evaluations.json`: machine-readable audit of the
  two parent vectors evaluated through the corresponding Full-model
  likelihood, including source hashes and parameter provenance. The original
  optimizer JSON files remain unchanged.
- `planned_model_contrasts.csv`: the planned categorical-enhancement and LPP
  AIC contrasts.
- `diagnostic_summary.csv`: every mean and interval plotted in the figure.
- `diagnostic_contrasts.csv`: category-recall and remembered-minus-forgotten
  LPP differences used to interpret the bars.
- `diagnostic_figure.svg`, `.pdf`, and `.png`: the tall two-column grouped-bar
  figure with one observed row and four scientifically selected prediction rows.
- `build_diagnostic_figure.R`: the single long-form ggplot facet specification
  used for all ten panels, row strips, repeated axes, and the shared legend.
- `diagnostic_figure_caption.qmd` and `diagnostic_figure_alt.txt`: publication-
  facing caption and alt text.
- `recall_rate_diagnostic_figure.*`: a shorter category-recall figure with the
  Negative-minus-Neutral difference printed for every row.
- `early_lpp_diagnostic_figure.*`: a shorter grouped-bar figure that places all
  four category-by-recall cells on one shared axis for each model row.
- `build_split_diagnostic_figures.R`: the shared ggplot source for those two
  alternatives. It reads `diagnostic_summary.csv` and leaves the original
  combined figure unchanged.
- `figure_plan.md` and `visual_language.md`: the research-question mapping and
  cross-figure visual specification used to design the diagnostic figure.
- `results_readout.qmd`: exact-value interpretation with scope cautions.
- `build_manifest.json`: input/output hashes and summarization settings.

## Summary definitions

Recall rate is the proportion of studied positions in a category that appear at
least once in a list's recall sequence. The legacy combined and split figures
use Early LPP mean-centered within each 20-item list. The manuscript-facing
diagnostic-effects table instead uses the original z-transformed Early-LPP
outcome from the empirical analysis. Its observed and predicted cell values are
first averaged within participant and then across participants. The fitted
model predictor remains within-list centered; displaying the derived EEG
outcome on its original scale does not change the fits.

Observed points are participant-level summaries. Their intervals are percentile
95% confidence intervals from 10,000 participant bootstrap samples using seed
`20260715`. Each prediction is summarized over 200 complete simulated datasets,
where one replicate contains simulated recalls for all 342 lists. Predicted
intervals are the central 95% of those replicate summaries and represent
simulation variability, not fitted-parameter uncertainty.

The simulations condition on the observed number recalled per list. The figure
therefore diagnoses which items enter the recalled set; it does not evaluate
termination or total recall. The LPP-by-memory-status bars were not separately
optimized and must be described as a derived diagnostic rather than independent
validation.

## Four-model figure subset

The numerical tables report all six approved fits. The diagnostic figure is
deliberately narrower because its job is to explain the planned contrasts, not
to duplicate the fit table. It shows:

1. Emotion-dependent LPP with categorical enhancement fixed at its neutral
   value;
2. categorical enhancement with no LPP term;
3. categorical enhancement with General LPP; and
4. categorical enhancement with Emotion-dependent LPP.

The first and fourth prediction rows isolate categorical enhancement, the
second and fourth isolate the addition of Emotion-dependent LPP, and the third
and fourth compare equal-complexity LPP mappings. The complete diagnostic
summaries for all six models remain in `diagnostic_summary.csv` and
`diagnostic_contrasts.csv`.

## Bound-sensitive comparison

Every fitted LPP slope was restricted to `[0, 0.2145443011]`.
`EEM_eCMR_LPP_General` fitted `kappa = 0.2079873`, near the upper bound. Because
no wider-bound sensitivity will be run, the General-versus-Emotion-dependent
comparison in the categorical-enhancement stratum is conditional on the
declared range. The exploratory Full-LPP table does not remedy its incomplete
optimization and cannot support a claim that a residual General component is
zero. It reports best evaluated candidates while retaining the raw optimizer
outputs.

No repository commit or GitHub issue update is part of this work unit.
