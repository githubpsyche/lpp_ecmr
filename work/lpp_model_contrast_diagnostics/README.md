# Contrast-first model diagnostics

This work unit is an alternative to `work/lpp_model_results/`, which remains
unchanged for collaborator comparison. It uses the same observed data and four
selected pooled model simulations but plots the scientific contrasts directly:

- Negative minus Neutral recall rate;
- Recalled minus Unrecalled Early LPP for Negative items; and
- Recalled minus Unrecalled Early LPP for Neutral items.

The figure contains no recall-status legend. Each diagnostic is named directly
in its column heading, and every model has one horizontal bar per contrast.

## Build

From the repository root, using the Python environment that provides `numpy`
and `h5py`:

```bash
python3 work/lpp_model_contrast_diagnostics/build_contrast_diagnostics.py
```

Figure generation requires R with `ggplot2` and `scales`. `pdftocairo`
converts the single ggplot PDF to SVG without changing its layout.

## Products

- `contrast_summary.csv`: estimates and correctly propagated intervals for all
  15 source-by-contrast cells.
- `contrast_diagnostic_figure.svg`, `.pdf`, and `.png`: the contrast-first
  publication prototype.
- `contrast_diagnostic_figure_caption.qmd` and
  `contrast_diagnostic_figure_alt.txt`: caption and accessibility text.
- `build_manifest.json`: input/output hashes and summary settings.

Observed intervals bootstrap participant-level contrasts. Predicted intervals
are calculated from contrasts within each of 200 complete simulated datasets.
They are not obtained by subtracting marginal confidence-interval endpoints.

No commit or GitHub issue update is part of this work unit.
