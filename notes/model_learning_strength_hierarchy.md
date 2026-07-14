# Model learning-strength hierarchy

This note records the working additive learning-strength rule from
[Issue #15](https://github.com/githubpsyche/lpp_ecmr/issues/15) and its
application to the model comparison set tracked in
[Issue #2](https://github.com/githubpsyche/lpp_ecmr/issues/2).

$$
\text{learning strength}_i
=
\underbrace{\phi_i}_{\text{ordinary position-dependent strength}}
+
\underbrace{\phi_{\mathrm{emot}}e_i}_{\text{categorical emotional increment}}
+
\underbrace{\kappa\,\widetilde{\mathrm{LPP}}_i}_{\text{general LPP contribution}}
+
\underbrace{\kappa_{\mathrm{emot}}e_i\widetilde{\mathrm{LPP}}_i}_{\text{additional emotional LPP contribution}}.
$$

Here, $e_i$ equals 1 for emotional items and 0 for neutral items.
$\widetilde{\mathrm{LPP}}_i$ is item $i$'s list-mean-centered Early LPP.

| Model | Learning destination | $+\phi_{\mathrm{emot}}e_i$ | $+\kappa\widetilde{\mathrm{LPP}}_i$ | $+\kappa_{\mathrm{emot}}e_i\widetilde{\mathrm{LPP}}_i$ |
|---|---|---:|---:|---:|
| CMR | Temporal | Fixed 0 | Fixed 0 | Fixed 0 |
| CMR + LPP (general) | Temporal | Fixed 0 | Fitted | Fixed 0 |
| CMR + LPP (emotional only) | Temporal | Fixed 0 | Fixed 0 | Fitted |
| CMR + LPP (full) | Temporal | Fixed 0 | Fitted | Fitted |
| EEM-CMR | Temporal | Fitted | Fixed 0 | Fixed 0 |
| EEM-CMR + LPP (general) | Temporal | Fitted | Fitted | Fixed 0 |
| EEM-CMR + LPP (emotional only) | Temporal | Fitted | Fixed 0 | Fitted |
| EEM-CMR + LPP (full) | Temporal | Fitted | Fitted | Fitted |
| Category-only eCMR | Source | Fixed 0 | Fixed 0 | Fixed 0 |
| Category-only eCMR + LPP (general) | Source | Fixed 0 | Fitted | Fixed 0 |
| Category-only eCMR + LPP (emotional only) | Source | Fixed 0 | Fixed 0 | Fitted |
| Category-only eCMR + LPP (full) | Source | Fixed 0 | Fitted | Fitted |
| EEM-eCMR | Source | Fitted | Fixed 0 | Fixed 0 |
| EEM-eCMR + LPP (general) | Source | Fitted | Fitted | Fixed 0 |
| EEM-eCMR + LPP (emotional only) | Source | Fitted | Fixed 0 | Fitted |
| EEM-eCMR + LPP (full) | Source | Fitted | Fitted | Fitted |

In every model, $\phi_i$ is present and its constituent primacy parameters are
fitted. In CMR-family models, the displayed learning strength governs
temporal-context-to-item learning. In eCMR-family models, it governs
source-context-to-item learning, while temporal-context-to-item learning
remains $\phi_i$.

“Fixed 0” means that the corresponding optional coefficient is fixed to its
additive neutral value. “Fitted” means that the coefficient is estimated. The
table summarizes learning-strength terms and their destination; it does not
represent every architectural difference between the CMR and eCMR families.
