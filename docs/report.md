# Sharing vs. separation in multilingual LLMs: what we tried and what we saw

Nov 3, 2026 · @Sathammai

## The question

When a multilingual model handles the same function in several languages, does it use one shared circuit or a separate one per language? We ran experiments on real models and on toy models to see which functions share, which separate, and why.

## Part 1: Real models

The model shares IOI, translation, negation, and agreement and gender for close languages. It separates word order everywhere, and gender and agreement for distant languages (HI worst).

**Setup.** Gemma-2 (2B, 9B, 27B, base and instruction-tuned) and BLOOM (1b7, 7b1). Each task was run across up to 7 languages (EN, FR, ES, DE, ZH, AR, HI, plus YO and SW on Gemma). Every task×language cell first had to pass an accuracy gate before any sharing measurement. We measured sharing with head ablation, head-output patching, residual-stream patching, representation probes, and later direction transfer (DAS-style).

### IOI (Gemma, Exp 10–14, 16–17)

- Early attempts to define language-specific subspaces and ablate them failed their own positive controls (Exp 10–12), so we dropped subspace erasure.
- Head ablation worked. EN-ranked IOI heads hurt FR and YO at every scale (retained logit-diff 0.17–0.35 at 2B, 0.54–0.74 at 27B). Sharing weakens with scale but persists.
- Head-output patching (Exp 14) is the cleanest result. EN head outputs patched into FR/YO runs recover about 100% of the effect at all scales. The heads write in a language-agnostic format.
- Representation and computation come apart (Exp 17). HI representations diverge from EN (+0.29 corrected cosine divergence) while the computation is still shared.
- Depth patching (Exp 16) found no clean layer where FR/DE split. HI was confounded by script.

### Agreement (Gemma, Exp 15b)

For FR/DE, cross-language compatibility rises with scale. The large values (4.32 at 27B FR) partly come from a small denominator, so the direction is reliable and the magnitude isn't. AR, SW and HI were lost to gate failures (tokenizer collisions, low accuracy).

### Task grid (BLOOM, Exp 18–22)

Every task×language cell was gated first. The cells that passed give:

| Task | Languages with a verdict | Result |
| --- | --- | --- |
| IOI | 7 | Shared |
| Translation | 6 | Shared |
| Negation | EN, FR, ES, ZH | Shared |
| Agreement | EN, FR, ES, HI | EN/FR shared, ES partial, HI weak |
| Gender | FR, ES, AR, HI | FR↔ES shared, AR weak, HI separated (cross-language patching hurts) |
| Word order | 7 | Separated (frac\_flip ≈ 0–0.2), including EN/FR/ES, which share the same order |
| Antonym | none | No sharing run was designed |
| ZH classifier, AR numeral polarity | none | No donor language, behavioural only |
| Factual recall | none | Never run |

The gender and agreement ordering is the same in both model sizes.

### Direction transfer (Gemma-9B-it, Exp 27–29)

The first two instruments (Exp 27 word order, Exp 28 gender) failed their gates. Exp 29 (DAS-style, necessity and sufficiency tests) gave:

- Gender: 11 of 12 cross-language cells transfer.
- Word order: 6 of 7 cells walled, with DE→EN ambiguous.
- The two directions are specific to their own feature.
- This is a single exploratory run on one instruction-tuned model.

### What the real-model data say

The one conflict is HI gender, which is separated on BLOOM but transfers on Gemma Exp 29. The instruments differ and we haven't resolved it. Across model sizes we saw no consistent trend: IOI sharing weakens with scale, agreement strengthens, and gender in HI strengthens (−0.24 to +0.32).

## Part 2: Synthetic models

The toy reproduces the word-order result: where linearizations differ, sharing costs the shared model and computation separates there. Where they don't, it shares.

**Aim.** Build small models where the number of languages, shortcut availability, language conflict and capacity are controlled directly, and see whether the sharing/separation split can be reproduced.

### Closed form and learnable allocation (Exp 1–2)

The closed-form allocation rule behaves as designed. In a small trained model with shared and private routes, languages peel off to private routes in the predicted order. We fixed three bugs along the way (gate collapse, a noise calibration error, a capacity-budget rounding error). Timing came out about 2× earlier than the formula predicts.

### PCFG agreement (Exp 3)

A language with a cheap shortcut needs less capacity, as expected. Four different designs for making languages conflict all failed. The model found a bypass each time (a majority heuristic, a lookup table absorbed by the MLP, or a rare-event override it never needed). The conflict cost was positive at some capacities but never shifted the capacity at which the task is solved.

### Conflicting gender (Exp 4)

Shared nouns with conflicting genders cost capacity (solved at embedding width 5 vs 4 with no conflict), but the required width was flat across overlap values 0 to 0.75, a switch rather than a gradient. A disjoint-vocabulary control was solved at width 3. That run is not saved in the repo.

### Paired grammars (Exp 5)

An Allen-Zhu-style two-grammar design was built with controls and never run (about 100–150 GPU-hours).

### Shared tree, different linearizations (Exp 6)

This is the only setup where the manipulation worked cleanly. One abstract tree is rendered three ways: L\_A as the anchor, L\_B with partial reversal, L\_C with full reversal.

- At small capacity, the shared model pays extra loss ordered by reordering distance (C > B > A), against a matched disjoint control.
- At d=96 the result depends on the architecture. Four-head runs show no extra loss for the shared model through 180K steps, and 12-head runs show it persisting across 2 seeds (L\_B +0.02–0.05, L\_C +0.06). Nobody has investigated why.

### Conversion cost (Exp 8, 6-cost)

We tested whether the cost of translating one rendering into another predicts sharing. It doesn't (Spearman −0.13), and the transfer prediction came out reversed in all 3 seeds. The measure was stable but wrong.

### Sharing measurements in the toy (Exp 7, 9b)

- A dual-role control with no real divergence showed the shared model beating dedicated ones at every learnable capacity.
- Interchange tests showed separation at the tree levels where the linearizations diverge and sharing where they don't. They were run on 3 trained checkpoints, evaluated separately, with no pooled statistics; thresholds were never frozen and there is no verdict. The claim rests on the first notebook version (v1); the revised instrument in v2 returned no usable numbers for its code-swap measurement.
- A terminal-conflict variant showed the shared model doing worse than a dedicated one at conflicted tokens (smoke test, single seed).
- The earlier sharing instruments (Pearson profiles, language probes, knockout correlation) all failed, from lexical leakage or outlier sensitivity.

### What the synthetic data say

The toy only varies word order. We never built a toy for gender or agreement that avoided a bypass, so the distance gradient has no synthetic counterpart.

## Where this leaves us

- **Established:** the task-dependent split on real models (IOI, translation and negation shared, word order separated), and its reproduction in the toy for word order.
- **Established as failing:** conversion cost as a predictor, a monotone decrease of sharing with scale, and every conflict design that gave the model a bypass.
- **Open:** whether the shared model's extra loss persists at larger capacity (the 4-head vs 12-head disagreement, and the unrun Allen-Zhu test); the HI gender conflict; and the gender/agreement distance gradient, which rests on n=24 cells.
- **Weak evidence:** the toy interchange results (3 checkpoints, no frozen thresholds or verdict), the single-seed Option B smoke test, and Exp 29 (one exploratory run).
- **Next:** rerun the interchange test with frozen thresholds, pooled statistics across checkpoints and a noise floor, resolve the 4-head/12-head gap, and replicate the distance gradient with more items and native-verified Hindi.
