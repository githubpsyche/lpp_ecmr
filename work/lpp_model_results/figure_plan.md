# Research-question-driven figure plan

The six-model fit and parameter tables provide the complete inferential record.
The diagnostic figure has a narrower explanatory purpose: it shows what the
focal mechanisms contribute to the two empirical patterns that motivated the
model comparison.

## Questions and intended demonstrations

| Research question | Formal evidence | Diagnostic demonstration |
|---|---|---|
| Does preferential categorical learning contribute? | All three absent-versus-present categorical-enhancement AIC contrasts | The recall-rate column shows that Emotion-dependent LPP without categorical enhancement does not recover the Negative recall advantage, whereas the corresponding combined model does. |
| Does Early LPP add information beyond categorical enhancement? | No-LPP versus LPP AIC contrasts in both categorical strata | The Early-LPP column shows that categorical enhancement without LPP recovers category-level recall but not the category-specific Remembered-Forgotten Early-LPP pattern. |
| Is one LPP slope better represented generally or emotion-dependently? | Equal-complexity General-versus-Emotion-dependent AIC contrasts | The Early-LPP column shows that General LPP separates Remembered and Forgotten items in both categories to differing degrees, whereas Emotion-dependent LPP concentrates the separation among Negative items. |

## Selected prediction rows

The figure contains one observed benchmark followed by four predictions:

1. `CategoryOnly_eCMR_LPP_EmotionalOnly`: Emotion-dependent LPP without
   preferential categorical enhancement;
2. `EEM_eCMR`: categorical enhancement without LPP;
3. `EEM_eCMR_LPP_General`: categorical enhancement plus General LPP; and
4. `EEM_eCMR_LPP_EmotionalOnly`: categorical enhancement plus
   Emotion-dependent LPP.

These rows make three controlled comparisons available around the final model:

- rows 1 and 4 isolate categorical enhancement;
- rows 2 and 4 isolate Emotion-dependent LPP; and
- rows 3 and 4 compare General with Emotion-dependent LPP at equal complexity.

The omitted baseline and General-LPP-without-categorical-enhancement models
remain in both numerical tables and the reproducible diagnostic CSV files.

## Layout

Use one five-row, two-column horizontal-bar facet grid rather than two figures
assembled after plotting. The left column shows Negative and Neutral recall
rates; the right column shows mean within-list-centered Early LPP for Remembered
and Forgotten items within each category. Both columns are generated from the
same long-form table and share the same model-row and category structure.

Model names occupy ggplot row strips immediately beside the grid. Metric names
occupy centered column strips, so separate vertical y-axis titles are
unnecessary. Horizontal scale ticks repeat in every model row. Bars use one
outline specification throughout, and the Remembered/Forgotten legend is
centered over the complete grid. Explanatory interval sublabels and panel
letters are omitted because those definitions belong in the caption.

Observed uncertainty is a percentile 95% participant-bootstrap confidence
interval. Prediction uncertainty is the central 95% interval across 200
complete simulated datasets. Both are retained, labeled separately, and
described in the caption because they have different interpretations. Model
intervals are not parameter-estimation uncertainty.

## Visual anchor

Empirical Figure 2B governs category, subsequent-memory, type, and theme
choices. Negative uses a red/salmon family; Neutral uses a grey family;
Remembered uses dark solid fill; Forgotten uses lighter fill and a dashed
outline. Helvetica, a white classic plotting field, and left/bottom axes track
the empirical package. Category and observed/predicted status are stated
directly; a centered legend states the Remembered/Forgotten mapping, so none of
these distinctions depends on colour.
