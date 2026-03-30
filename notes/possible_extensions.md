# Possible extensions to the model comparison

This note considers four questions not addressed by the current model set but relevant to the paper's argument or its limitations. Each may inform Discussion framing, future modelling work, or both.

The paper has three aims: (1) LPP is individually reliable, (2) LPP predicts emotional memory at the trial level (but not between subjects), (3) incorporating LPP into eCMR improves population-level fit. The core finding is that emotional context architecture (eCMR) is the dominant driver, with LPP providing a modest additive gain at the population level that doesn't reliably help at the individual level.

This note considers four questions not addressed by the current model set but relevant to the paper's argument or its limitations. Each may inform Discussion framing, future modelling work, or both.

What the 24 models address
Termination: Well-covered. Three stop conditions test whether the forced-output-length simplification matters.

Architecture ladder: CMR → LPP_CMR → eCMR → LPP_eCMR. Tests whether LPP can substitute for or supplement emotional context.

Strength baselines: Tests whether context dynamics matter at all vs pure encoding strength + Luce choice. This is a floor that makes the CMR/eCMR results more interpretable — without it, a reader could wonder if a simpler mechanism explains the data.

What's missing — scientific questions the paper raises but the models don't test

## 1. Semantic confound (item 3, but broader). 

Neutral items contain semantic clusters. eCMR treats all neutral items identically. If semantic clustering drives some of the recall structure in neutral lists, eCMR's emotional context pathway could be getting credit for patterns that are actually semantic. This is a confound, not just a limitation footnote. Would need embeddings + semantic CMR variants.

Evaluation: The paper's goal is to show that LPP improves eCMR's account of emotional memory. The model comparison evidence for this is the ΔAIC between eCMR and LPP-eCMR. If there's unexplained semantic clustering variance in the data, it's noise from the model's perspective — it hurts the likelihood equally for all models. That doesn't change the relative ranking or the ΔAIC between models. Adding semantic similarity would improve absolute fit for all models but wouldn't necessarily change which model wins or by how much. So... it might not matter much for the paper's core claim. The semantic gap is a limitation of the models' absolute fit, but it's not obviously a confound for the relative comparisons that the paper relies on.

The one scenario where semantic models would change the paper's conclusions: if the emotional memory advantage (emotional items recalled more than neutral) is partly driven by emotional items being more semantically coherent as a category than the neutral items. If all negative items share a "threat" semantic cluster while neutral items are mixed (some related, some not), then eCMR's emotional context pathway might be partly capturing what is really a semantic coherence advantage. A semantic model would attribute that to semantic similarity rather than emotional context, potentially reducing eCMR's advantage over semantic CMR.

Is there reason to think the negative items are more semantically coherent as a group than the neutral items? Yes. Negative items often cluster around a smaller set of themes (e.g., violence, illness, death) than neutral items, which can be more heterogeneous. This could give negative items a stronger semantic cluster that boosts recall. If eCMR is picking up on this semantic clustering as part of its emotional context mechanism, then the paper's claim that emotional context dynamics drive the recall advantage for negative items would be confounded by semantic coherence.

Still, we don't have the means to evaluate without richer semantic data per item. Without per-item semantic features, we can only use the binary category membership (emotional, related neutral, unrelated neutral) as a proxy for semantic similarity. That gives us a coarse 3-level categorical similarity structure, not real inter-item semantic distances. So the semantic model question is real but we can't properly test it without richer per-item data (word embeddings, LSA vectors, etc.). It stays as an acknowledged limitation rather than something we can evaluate in this revision.

## 



2. Isolating architecture from encoding modulation (item 12, but scientifically important). The eCMR vs CMR comparison bundles dual context + emotion-modulated encoding. A reader naturally asks: is it the emotional context that matters, or is it just that emotional items get stronger encoding? Missing models:

Single-context + emotion encoding (tests encoding modulation without architecture change)
eCMR with emotion_scale=0 (tests architecture without encoding modulation)
3. Alternative LPP parameterizations. LPP currently only modulates φ_emot (encoding strength for emotional items). But LPP could plausibly modulate:

Temporal context drift rate (items with higher LPP get more context update)
Retrieval dynamics (higher LPP items are more easily reinstated) The paper doesn't test these alternatives, so the reader can't know if the encoding-strength parameterization is the right one or just the one that was tried.
4. General-LPP (items 6/16, but stands on its own scientifically). Is LPP's memory-predictive value truly emotion-specific, or does the model just not let it help neutral items? The GLMM says the effect is emotion-specific, but the GLMM is on the same data. A model comparison where LPP can help all items would be the proper test.

5. The between-subjects paradox. The paper's most surprising finding: strong trial-level LPP→memory coupling, but no between-subjects correlation of LPP advantage with memory advantage. None of the 24 models interrogate this. A hierarchical or subject-grouping analysis (e.g., do subjects with reliable LPP effects show better LPP-eCMR fits?) would address this directly.

6. Cross-validation or held-out evaluation. All fits and evaluations are on the same dataset. The "genuine generalization" claim (that LPP-eCMR predicts recalled-vs-unrecalled LPP splits without being fit to them) is the paper's strongest validation, but a reader would want to see held-out subjects or leave-one-list-out cross-validation to know the population-level advantage isn't overfitting.

What might be over-represented
The 12 Strength variants (half the model set) will almost certainly be dominated by CMR variants. They serve a useful floor function but 12 models is heavy for that role — 4 (no stop, PositionalStop) would probably suffice unless the Strength × stop interaction tells an interesting story.

Priority ranking for missing comparisons
Mechanism isolation (enable single-context emotion model + eCMR with no encoding modulation) — cheapest, directly addresses the paper's strongest claim
Cross-validation — the infrastructure exists (cross_validation.ipynb template in lpp_ecmr), just needs to be run
General-LPP — requires code change but is the clean empirical answer to Deborah's circularity concern
Semantic models — requires data prep but addresses a real confound
Alternative LPP parameterizations — would require new model code
Between-subjects analysis — analysis question, not a model variant