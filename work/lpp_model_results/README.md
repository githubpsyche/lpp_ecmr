# Six-model pooled result package

This package converts the pooled mixed-list fits and simulations in
`work/lpp_model_comparison/` into the numerical tables and diagnostic summaries
used by the manuscript and its selected figures. It does not modify the source
fit or simulation artifacts.

The comparison crosses the emotion-based learning boost (absent/present) with
the LPP-based learning boost (none/all items/negative items only), producing six
source-context models.

## Build

From the repository root:

```bash
python3 work/lpp_model_results/build_results.py
```

The builder checks the selected fits, declared parameter bounds, HDF5 shapes,
simulation storage order, list count, participant count, simulation count, and
within-list LPP centering before producing any summaries.

## Products

- `model_comparison.csv` and `model_comparison_table.qmd`: six-model NLL, AIC,
  and delta-AIC results.
- `parameter_estimates.csv` and `parameter_table.qmd`: complete pooled
  parameter estimates, fitted ranges, fixed values, and the
  implementation-to-manuscript mapping.
- `planned_model_contrasts.csv`: planned emotion-based-learning and LPP AIC
  contrasts.
- `diagnostic_summary.csv`: observed and predicted means and intervals consumed
  by the selected empirical-versus-predicted figures.
- `diagnostic_contrasts.csv`: negative-minus-neutral recall and
  remembered-minus-forgotten LPP contrasts.

## Summary definitions

Recall rate is the proportion of studied positions in a category that appear at
least once in a list's recall sequence. The diagnostic summaries use Early LPP
mean-centered within each 20-item list. The matched Early-LPP figure in
`work/lpp_model_prediction_grids/` reports the derived EEG outcome on its
original z-transformed scale.

Observed points are participant-level summaries. Their intervals are percentile
95% confidence intervals from 10,000 participant bootstrap samples using seed
`20260715`. Each prediction is summarized over 200 complete simulated datasets,
where one replicate contains simulated recalls for all 342 lists. Predicted
intervals are the central 95% of those replicate summaries and represent
simulation variability, not fitted-parameter uncertainty.

The simulations condition on the observed number recalled per list. The
diagnostics therefore concern which items enter the recalled set; they do not
evaluate recall termination or total recall. The LPP-by-memory-status summaries
were not separately optimized and are derived diagnostics rather than
independent validation.

Every fitted LPP slope was restricted to `[0, 0.2145443011]`. The generated
parameter table reports the estimates against that declared range.
