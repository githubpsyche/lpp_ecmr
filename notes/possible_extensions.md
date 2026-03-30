# Possible extensions to the model comparison

The paper's modelling section asks whether incorporating trial-level LPP amplitude into the eCMR architecture improves its account of emotional free recall. A preceding GLMM analysis establishes that early LPP predicts subsequent recall for emotional items but not neutral items, motivating an emotion-gated LPP parameterization.

The manuscript's model comparison currently evaluates four models:

- **CMR** (k=9): temporal-context-only baseline, no emotion or LPP
- **CMR+LPP** (k=10): single context, LPP modulates encoding for all items, no emotion category information
- **eCMR** (k=10): dual temporal/emotional context, emotion category modulates encoding via φ_emot, no LPP
- **LPP-eCMR** (k=11): eCMR + trial-level LPP modulates encoding for emotional items only

All four are fit using set-permutation likelihood conditioned on observed output length (forced stopping). The core finding is that the emotional context architecture (eCMR) is the dominant driver of fit improvement, with LPP providing a modest additional gain at the population level that does not reliably improve individual subject-level fits.

This note considers five extensions not addressed by this comparison but relevant to the paper's argument or its limitations.

## Termination policy

Right now we go out of our way to avoid including recall termination in our loss function. But ideally recall counts should fall naturally out of our model architecture. So we are motivated to explore Positional and Ratio termination policies so we can simplify the manuscript. If we can simply find that we achieve the same basic LPP-eCMR > eCMR results with the ratio termination rule, we'll be in the best spot.

## Context-independent baseline

Since the data lacks recall order, a particularly sensitive question is whether using eCMR (a model primarily of the organization of free recall!) for this evaluation is even well-motivated. Including pure strength-based models matched to eCMR's architecture except excluding a temporal context representation is a good way to validate our evaluation.

## Mechanism isolation

The CMR-to-eCMR jump changes two things at once: it adds a dual emotional context pathway AND it lets emotion category modulate encoding strength via φ_emot. So when eCMR beats CMR, we can't say which of those two changes is doing the work. A single-context model with emotion encoding modulation would test whether the encoding boost alone is enough. An eCMR variant with emotion_scale=0 would test whether the dual context pathway helps even without the encoding boost.

## General LPP

LPP-eCMR only lets LPP modulate encoding for emotional items. This restriction comes from the GLMM finding that LPP predicts recall for emotional items but not neutral ones. But the GLMM is on the same dataset, so the restriction is partly circular. A variant where LPP modulates all items would test whether the restriction is actually helping or just baking in the GLMM result. This is currently blocked by a code-level entanglement between the LPP and φ_emot channels in the eCMR implementation.

## Semantic structure

None of the models represent pre-experimental semantic similarity between items. The TalmiEEG neutral items include semantic clusters that the models collapse. Ideally we'd include semantic similarity to clarify its contribution to model performance and improve absolute fit. But we lack per-item embeddings, and category membership alone is too coarse to be useful since it would largely duplicate what eCMR's emotional context already represents.

For the core eCMR-vs-LPP-eCMR comparison, this is probably not a problem: unexplained semantic variance hurts all models roughly equally, so relative rankings should be unaffected. A more open question is whether negative items are more semantically coherent as a category than neutral items. If so, eCMR's emotional context pathway could partly capture a semantic advantage, which would inflate the CMR-vs-eCMR comparison rather than the LPP comparison. We haven't checked whether the TalmiEEG stimuli show this pattern. The mechanism isolation models from section 3 might partially address this concern by separating the encoding-strength contribution from the architecture contribution.