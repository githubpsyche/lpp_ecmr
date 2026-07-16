# Visual language for the Issue #10 prototype

The prototype uses independent, redundant channels for category, recall status,
and observed/predicted status.

| Meaning | Encoding |
|---|---|
| Negative category | Red/salmon family based on `#D95F5F`; direct row label |
| Neutral category | Grey family based on `#666666`; direct row label |
| Remembered | Dark, solid category fill |
| Forgotten | Lighter category fill with the same dark, dashed outline |
| Observed | Top row with the direct label `Observed data` |
| Predicted | Separate row labeled with the manuscript-facing model variant name |
| Empirical uncertainty | Black capped percentile 95% participant-bootstrap interval |
| Simulation variability | Black capped central 95% interval across simulated datasets |

Category is never conveyed by colour alone: position and the words `Negative`
and `Neutral` repeat the distinction. Recall status is never conveyed by colour:
solid versus lighter/dashed fill distinguishes `Remembered` and `Forgotten`.
Observed and predicted quantities are stated directly rather than inferred from
styling. Negative and Neutral are repeated in every model row. A compact
Remembered/Forgotten legend explains the fill/outline encoding without repeating
four long labels under every row. It is centered over the complete facet grid.
Interval definitions are reserved for the caption rather than repeated as row
sublabels.

This mapping is anchored to empirical Figure 2B, which uses red/pink for
Negative items, grey/black for Neutral items, and lightness to distinguish
subsequent-memory cells. The model figure retains those semantic colour
families while replacing pure black with dark grey and adding outline redundancy.
It uses Helvetica, a white `theme_classic`-like field, and left/bottom axes to
track the empirical figures. Panel letters are omitted because the column
titles already identify the two displays. It does not reuse Figure 4's
purple/green mapping because that palette distinguishes Early versus Late time
windows, a variable absent from the model diagnostic.

The fill/outline and position encodings preserve the comparisons in grayscale.
Final cross-package harmonization with the mechanism figure remains subject to
review under Issue #21.
