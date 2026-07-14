# Model comparison summary: per-subject and group-level fitting

## Per-subject fitting (N=38)

Each subject's 9 free-recall lists were fit independently. For nested model pairs, the child model's per-subject NLL is floored at the parent's value to correct for optimizer failures (23 subject-model entries floored: 13 for CMR+LPP, 10 for LPP-eCMR).

| Model | k | Mean AIC | Total AIC | AICw (total) | Mean AICw (per-subject) |
|:------|:--|:---------|:----------|:-------------|:------------------------|
| CMR | 9 | 448.02 | 17024.76 | ~0 | 0.21 |
| CMR+LPP | 10 | 447.69 | 17012.05 | ~0 | 0.17 |
| eCMR | 10 | 443.96 | 16870.39 | .003 | 0.29 |
| LPP-eCMR | 11 | 443.66 | 16859.03 | .997 | 0.33 |

**CMR vs CMR+LPP.** The GLMM finds that LPP predicts recall for emotional items (β=.14, p<.001) but not neutral items (β=.01, p=.488). CMR+LPP tests whether LPP improves prediction when introduced into CMR through its temporal-pathway encoding mechanism, in a single-context architecture with no emotional context layer or category labels. Mean ΔAIC: 0.33. CMR+LPP does not improve on CMR. With 9 lists per subject and 9-10 free parameters, the data do not support estimating subject-specific LPP augmentation reliably.

**eCMR vs CMR.** eCMR adds an emotional context dimension and category-based encoding boost (same k as CMR+LPP). Mean ΔAIC: 4.06 favoring eCMR. eCMR wins in 23/38 subjects.

**LPP-eCMR vs eCMR.** LPP-eCMR adds LPP-modulated encoding for emotional items only (k=11, one extra parameter over eCMR). Mean ΔAIC: 0.30 favoring LPP-eCMR. LPP-eCMR wins in 16/38 subjects.

**CMR+LPP vs eCMR.** Mean ΔAIC: 3.73 favoring eCMR. eCMR wins in 26/38 subjects. LPP through the temporal pathway alone cannot replace category labels.

## Group-level fitting

Per-subject fitting is a no-pooling approach where each subject gets a full parameter vector. Complete pooling fits a single shared parameter vector to all 342 trials. This is a different model of subject heterogeneity, not just the same comparison with less noise. In particular, each additional parameter costs 2 AIC units under pooling versus 76 (2 × 38 subjects) in the no-pooling analysis.

| Model | k | -LL | AIC | ΔAIC |
|:------|:--|:----|:----|:-----|
| CMR | 9 | 8317.75 | 16653.51 | +101.42 |
| CMR+LPP | 10 | 8308.85 | 16637.71 | +85.62 |
| eCMR | 10 | 8283.13 | 16586.25 | +34.16 |
| LPP-eCMR | 11 | 8265.05 | 16552.09 | 0 (best) |

**CMR vs CMR+LPP.** Under complete pooling, CMR+LPP improves substantially on CMR (ΔAIC = 15.80, lpp_main_scale = 56.4). At the individual-subject level, adding LPP does not yield a reliable improvement when each subject is fit separately from only 9 lists. This pattern suggests that LPP carries a population-level signal that is too weak to estimate robustly in subject-specific fits with the current amount of behavioral data.

**eCMR vs CMR and CMR+LPP.** eCMR dominates both single-context models under pooling (ΔAIC = 67.26 over CMR, 51.46 over CMR+LPP), consistent with the per-subject results.

**LPP-eCMR vs eCMR.** LPP-eCMR beats eCMR by 34.16 AIC points under pooling. In the per-subject comparison, the mean ΔAIC was 0.30 and the AICw conclusion depended on nesting flooring. Under complete pooling, the LPP advantage is unambiguous.
