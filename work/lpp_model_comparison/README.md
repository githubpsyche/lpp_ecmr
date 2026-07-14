# Pooled Early LPP model comparison

This work package contains the prepared pooled fits for the 16-model hierarchy
tracked in [Issue #2](https://github.com/githubpsyche/lpp_ecmr/issues/2). The
hierarchy crosses:

- temporal-only CMR versus source-context eCMR;
- absence versus presence of categorical emotional scaling; and
- no LPP effect, a general LPP effect, an emotional-only LPP effect, or both
  LPP effects.

The executable source of truth is
[`model_comparison_registry.py`](../../lpp_ecmr/model_comparison_registry.py).
The learning-strength composition is implemented in
[`learning_strength.py`](../../lpp_ecmr/models/learning_strength.py). Issue
[#16](https://github.com/githubpsyche/lpp_ecmr/issues/16) specifies categorical
emotional scaling, and Issue
[#19](https://github.com/githubpsyche/lpp_ecmr/issues/19) specifies the
log-linked LPP rule.

## Learning-strength rule

For the context-to-item pathway modified by a given model, learning strength is

$$
S_i^X
=
\underbrace{B_i^X}_{\text{ordinary pathway baseline}}
\underbrace{\left[1+(\phi_{\mathrm{emot}}-1)e_i\right]}_{\text{categorical emotional scaling}}
\underbrace{
\exp\!\left(
\kappa z_i+
\kappa_{\mathrm{emot}}e_i z_i
\right)
}_{\text{LPP modulation}},
$$

where

$$
B_i^T=\phi_i,
\qquad
B_i^S=\omega\phi_i.
$$

Here:

- $X=T$ denotes temporal-context-to-item learning and $X=S$ denotes
  source-context-to-item learning.
- $\phi_i$ is the ordinary position-dependent primacy strength.
- $\omega$ is the fitted source-learning scale relative to temporal learning.
  Thus, $\omega=1$ gives equal ordinary temporal and source learning.
- $e_i=1$ for a negative (emotionally negative) item and $e_i=0$ for a neutral
  item.
- $z_i$ is the stored, pre-stimulus-standardized Early LPP amplitude after
  within-list mean centering:

  $$
  z_i
  =
  \mathrm{EarlyLPP}_i
  -
  \overline{\mathrm{EarlyLPP}}_{\operatorname{list}(i)}.
  $$

- $\phi_{\mathrm{emot}}$ is a categorical multiplier with neutral value 1.
- $\kappa$ is the neutral-item slope on log learning strength.
- $\kappa_{\mathrm{emot}}$ is the additional emotional-item log-learning
  slope. Emotional items therefore have total LPP slope
  $\kappa+\kappa_{\mathrm{emot}}$.

The link keeps learning strength positive without flooring. The terms are
additive on the log-learning-strength scale:

$$
\log S_i^X
=
\log B_i^X
+
e_i\log\phi_{\mathrm{emot}}
+
\kappa z_i
+
\kappa_{\mathrm{emot}}e_i z_i.
$$

## Implemented hierarchy

| Registry model | Modified pathway | Ordinary baseline | $\phi_{\mathrm{emot}}$ | $\kappa$ | $\kappa_{\mathrm{emot}}$ | Free parameters |
|---|---|---|---:|---:|---:|---:|
| `CMR` | Temporal | $\phi_i$ | Fixed 1 | Fixed 0 | Fixed 0 | 9 |
| `CMR_LPP_General` | Temporal | $\phi_i$ | Fixed 1 | Fitted | Fixed 0 | 10 |
| `CMR_LPP_EmotionalOnly` | Temporal | $\phi_i$ | Fixed 1 | Fixed 0 | Fitted | 10 |
| `CMR_LPP_Full` | Temporal | $\phi_i$ | Fixed 1 | Fitted | Fitted | 11 |
| `EEM_CMR` | Temporal | $\phi_i$ | Fitted | Fixed 0 | Fixed 0 | 10 |
| `EEM_CMR_LPP_General` | Temporal | $\phi_i$ | Fitted | Fitted | Fixed 0 | 11 |
| `EEM_CMR_LPP_EmotionalOnly` | Temporal | $\phi_i$ | Fitted | Fixed 0 | Fitted | 11 |
| `EEM_CMR_LPP_Full` | Temporal | $\phi_i$ | Fitted | Fitted | Fitted | 12 |
| `CategoryOnly_eCMR` | Source | $\omega\phi_i$; $\omega$ fitted | Fixed 1 | Fixed 0 | Fixed 0 | 11 |
| `CategoryOnly_eCMR_LPP_General` | Source | $\omega\phi_i$; $\omega$ fitted | Fixed 1 | Fitted | Fixed 0 | 12 |
| `CategoryOnly_eCMR_LPP_EmotionalOnly` | Source | $\omega\phi_i$; $\omega$ fitted | Fixed 1 | Fixed 0 | Fitted | 12 |
| `CategoryOnly_eCMR_LPP_Full` | Source | $\omega\phi_i$; $\omega$ fitted | Fixed 1 | Fitted | Fitted | 13 |
| `EEM_eCMR` | Source | $\omega\phi_i$; $\omega$ fitted | Fitted | Fixed 0 | Fixed 0 | 12 |
| `EEM_eCMR_LPP_General` | Source | $\omega\phi_i$; $\omega$ fitted | Fitted | Fitted | Fixed 0 | 13 |
| `EEM_eCMR_LPP_EmotionalOnly` | Source | $\omega\phi_i$; $\omega$ fitted | Fitted | Fixed 0 | Fitted | 13 |
| `EEM_eCMR_LPP_Full` | Source | $\omega\phi_i$; $\omega$ fitted | Fitted | Fitted | Fitted | 14 |

“Fixed 1” is the neutral value for categorical emotional scaling. “Fixed 0”
is the neutral value for either log-LPP slope.

Every model contains nine shared free parameters. A source-context model adds
two: source encoding drift and the relative source-learning scale $\omega$.
Categorical emotional scaling adds $\phi_{\mathrm{emot}}$, and each enabled LPP
slope adds one parameter. This gives the free-parameter counts shown in the
table. Source recall drift is fixed to 1 in every source-context model.

In temporal-only models, the displayed rule modifies temporal-context-to-item
learning. In source-context models, it modifies source-context-to-item learning;
temporal-context-to-item learning remains $\phi_i$. “Category-only” source
models still represent negative and neutral source categories, but they fix
$\phi_{\mathrm{emot}}=1$, so ordinary source-learning strength does not differ
by category.

## Parameter bounds

The current registry uses:

$$
\omega\in[\epsilon,10],
\qquad
\phi_{\mathrm{emot}}\in[\epsilon,10]
\quad\text{when fitted},
$$

and, whenever an LPP slope is fitted,

$$
\kappa,\ \kappa_{\mathrm{emot}}
\in
\left[
0,
\frac{\log 10}{10.732445846153846}
\right]
\approx
[0,0.2145443011].
$$

The nonnegative LPP bounds encode the directional hypothesis reported in the
manuscript. The upper bound allows either LPP term to contribute at most a
tenfold multiplier at the largest observed positive $z_i$. In a full model,
the two terms can therefore combine to contribute at most a $100$-fold
multiplier for an emotional item at that value.

The current interval for $\phi_{\mathrm{emot}}$ permits both attenuation
($\phi_{\mathrm{emot}}<1$) and enhancement
($\phi_{\mathrm{emot}}>1$), despite the `EEM` registry label. If the scientific
intention is to permit enhancement only, its lower bound must be changed to 1
before fitting.

## Work-package contents

- `render_fits.ipynb` prepares the model-specific fitting notebooks.
- `fit_<model>.ipynb` is the cluster-ready notebook for one registry model.
- `fit_grid.csv` records each model's free and fixed parameters.
- `run_manifest.json` records the expected outputs and current preparation
  state.
- `unit_checks.csv` records the deterministic pre-fit validation results.
- `review.ipynb` validates and aggregates returned results only after all
  expected products are present.

This README summarizes the learning-strength hierarchy and the parameter
differences needed to interpret its counts. Other shared fitting choices—such
as pooled fitting, the set-permutation likelihood, forced stopping, and
optimizer settings—remain defined by the executable registry and run manifest.
