# Cost-Theoretic Sharing vs. Separation in Multilingual LLMs

## Consolidated Experiment Report — v2

**Author**: Sathammai (IIT Delhi LCS2, Research Intern)
**For**: Anwoy (Mentor)
**Date**: November 2026
**Repo**: `sat1509/pdt` — 73 Jupyter notebooks

---

## Changelog (v1 → v2)

| # | Section | What changed | Why |
|---|---------|-------------|-----|
| 1 | 2.6.1 (Exp 6) | Replaced "ε fades to ~0 by d=96" with full presentation of BOTH h4 and h12 results. Added step-matched table (2 seeds, h12). Rewrote verdict. | Cell 47 of main notebook recomputes eps after step-matching: L_B=+0.023, L_C=+0.060 (seed 1) and L_B=+0.046, L_C=+0.064 (seed 2). h12 and h4 configurations disagree; report v1 used only h4. |
| 2 | 1.2.2 (Exp 15b) | Added denominator analysis for compat > 1. Quoted notebook's theory prediction. Corrected AR from "gate failure" to "tokenizer collision (all verb pairs dropped)." | compat >> 1 at 27B FR reflects tiny gap (retained EN-heads→lang = +0.77), not genuine enhancement. No cost_derivation.md exists; prediction is from notebook markdown only. |
| 3 | 2.3 (Exp 3) | Changed "Uniformly failed" to "mechanism partially engages but does not produce differential N*." | eps_cost IS positive at cap=6-8 (+0.063, +0.070). This is partial engagement, not uniform failure. |
| 4 | 2.7.2 (Exp 7-ser) | Added evidence-level caveat. *(Corrected in repo cleanup: an earlier version said "single-seed, bp_h4_d96_s0 only".)* | Both notebook versions evaluate three checkpoints (bp_h4_d96_s0, s7, s13) separately, with no pooled statistics, frozen thresholds or verdict. |
| 5 | 2.9 (Exp 9b) | Added single-seed caveat. | seeds=[0], 9 models total. |
| 6 | 3.4 (Falsified/Unresolved) | Restructured: framework verdict now "specific mechanisms failed; framework-level test incomplete." | h12 data shows A2 is not clearly violated. "Not rescuable" was based partly on h4-only Exp 6 reading. |
| 7 | 3.7 (Uncertainty #2) | Removed false claim "does not recompute ε after matching." Replaced with accurate characterisation of the h4/h12 discrepancy. | Cell 47 explicitly performs this recomputation. |
| 8 | Executive framing | Restructured: empirical findings first (task-dependent asymmetry), framework verdict second. | Per user request. |
| 9 | 2.6.2 (Exp 6-cost) | Corrected "3 seeds" to "2 seeds visible in saved output." | Only seeds 0 and 1 a(g) values are in notebook output. |
| 10 | 1.5.5 (Exp 29) | Clarified model is 9B-it specifically; no base or 27B run exists. | Confirmed from cell 3: Gemma2ForCausalLM. |

---

## Executive Summary

### What the experiments found

Across 73 notebooks probing real LLMs (Gemma-2, BLOOM) and synthetic toy models, **five empirical findings are robust**:

1. **IOI circuits are shared** across languages and scales. Head-output patching (Exp 14) gives compatibility ≈ 1.0 at all model sizes — the cleanest result in the project.

2. **Word-order computation is language-specific.** Converging evidence from BLOOM patching (Exp 22), CKA (Exp 25), and DAS direction transfer (Exp 29): 6/7 cross-language order cells hit a WALL.

3. **Gender directions transfer cross-linguistically.** 11/12 gender cells = TRANSFER in Exp 29 (DAS-lite, Gemma-2-9B-it). Additive, necessary, specific, and crosses language families (FR→AR R = 0.94).

4. **Representation and computation are dissociable.** HI representations diverge (Exp 17, corrected cosine = +0.290) while computation is shared (Exp 14, compat ≈ 1.0).

5. **τ (shortcut richness) works.** In every synthetic test (Exp 1, 2, 3), high-τ languages specialise earlier.

The **task-dependent sharing asymmetry** — gender shared, word order separated, IOI shared, translation shared, agreement shared — is the project's most reliable high-level finding. It does not require the N*/ε framework to state.

### What happened to the framework

The N*/ε framework makes three testable claims: (a) ε is an irreducible floor that does not shrink with capacity (A2); (b) conversion cost a(g) governs N*; (c) sharing decreases with scale via an N* survival curve.

- Claim (b) is **falsified**: Spearman(a, gain) = −0.13 (Exp 8); a(L_C) < a(L_B) while ε_C > ε_B (Exp 6-cost).
- Claim (c) is **contradicted for agreement**: sharing increases with scale in Exp 15b (compat rises from 2B to 27B for FR and DE).
- Claim (a) is **unresolved and configuration-dependent**: at d=96, h4 runs show ε ≈ 0 (A2 violated) while h12 runs show ε > 0 across 2 seeds (A2 not clearly violated). The best-designed test (Exp 5, Allen-Zhu CFGs) was never completed.

**Assessment**: specific mechanisms predicted by the framework (a(g) → N*, monotonic scale decrease) are falsified. Whether ε exists as an irreducible floor (the framework's core assumption A2) is genuinely unresolved — not confirmed, not refuted, but untested at the required standard. The framework-level test is incomplete.

---

## Framing

This project asks: **when does a multilingual model build shared circuits across languages, vs. language-specific (separated) circuits?**

The theoretical framework posits a cost-theoretic answer parameterised by three quantities:

- $N$ — model capacity (width, depth, or parameters)
- $\tau$ — **shortcut richness**: how cheaply a language-specific shortcut substitutes for shared computation. High $\tau$ means a cheap private route exists.
- $\varepsilon$ — **anchor-distance quality floor**: irreducible representational penalty the shared space imposes on distant languages. Key assumption **A2**: $\varepsilon$ does NOT shrink with capacity.

The central prediction is a **capacity threshold** $N^*(\tau, K, \varepsilon)$ below which sharing wins and above which separation wins, with closed form:

$$N^*(\tau, K, i) = \frac{\gamma \, K^{\phi_i}}{\gamma - 1/K}, \qquad \gamma = (1 - \tau)^{1/\beta}$$

where $K$ = number of languages and $\phi \sim 0.11$ (sublinear cost scaling). The aggregate prediction: **sharing decreases with scale** across a population of functions with distinct $N^*$ values, producing a survival curve of shared functions.

---

# PART 1: Natural-Data Experiments

These experiments probe real LLMs (Gemma-2 family, BLOOM) for evidence of shared vs. separated circuits, using causal interventions (head ablation, patching, direction transfer) and representational probes.

---

## 1.1 IOI Circuit Sharing: Gemma-2 (Exp 10–14)

### 1.1.1 Subspace Separability Arc (Exp 10 → 11 → 12)

**Question**: Can we define language-specific subspaces of the residual stream and show that IOI circuits occupy separable subspaces?

This is a four-experiment arc of progressive instrument debugging.

**Exp 10** (`Exp10_Gemma_IOI_Representation.ipynb`): Gemma-2 2B, EN/FR/YO. Variance-PCA subspace definition → head ablation + subspace ablation. Head ablation confirmed sharing (EN-ranked heads hurt FR: retained = 0.376, hurt YO: retained = 0.488). But subspace ablation showed **no effect** (all retained $\approx$ 1.0).
- **Verdict**: *Instrument inadequate* — variance-PCA subspaces don't capture IOI-relevant directions.

**Exp 11** (`Exp11_Gemma_IOI_CausalSubspace.ipynb`): Switched to gradient-defined subspaces ($\partial \text{logit-diff} / \partial \text{residual}$). Mean-ablation at END position only. **Positive control failed for all $K$ values**: own-subspace ablation retained $> 1.0$ for all languages (EN: 1.008–1.015 across $K = 4, 8, 16, 32$). The mean-ablation at a single position gets healed by downstream layers.
- **Verdict**: *Instrument inadequate* — single-position mean-ablation gets healed.

**Exp 12** (`Exp12_Gemma_IOI_Erasure.ipynb`): Projection erasure through all remaining layers. Partial fix: FR own retained = 0.572, YO own retained = 0.500 (both PASS threshold $< 0.8$). EN own retained = 0.770–0.919 (FAIL). Cross-language ratios where positive control passed: FR→YO $\approx$ 0.85–0.95 (moderate sharing).
- **Verdict**: *Partial instrument fix.* EN still fails positive control. **Design flaw**: EN name count (45) >> FR (34) >> YO (20); gradient subspace dimension is inflated proportional to name count.

**Exp 13** (`Exp13_Gemma_IOI_Scale.ipynb`): Fixed confound with equal name counts (20 per language). Full run on Gemma-2 2B, 9B, 9B-it, 27B-it.

Head ablation (EN-ranked heads → target, retained logit-diff):

| Model | EN | FR | YO |
|-------|------|------|------|
| 2B | 0.169 | 0.293 | 0.354 |
| 9B | 0.219 | 0.373 | 0.486 |
| 9B-it | 0.261 | 0.436 | 0.559 |
| 27B-it | 0.538 | 0.651 | 0.740 |

All values well below random-baseline retained $\approx 1.0$: EN-ranked heads consistently hurt other languages. **Sharing decreases with scale** (retained rises from 2B to 27B), but sharing persists at all scales tested.

Subspace erasure instrument remained unreliable (positive control failures persisted even with equal name counts). The subspace-separability measurement track was **effectively abandoned** after this experiment.
- **Verdict for Exp 13**: Head ablation is valid and robust. Subspace erasure instrument never passed gate.

### 1.1.2 Head-Output Patching (Exp 14)

`Exp14_HeadOutput_Patching.ipynb` — Gemma-2 2B/9B/9B-it/27B-it. Full run.

**Question**: Do IOI heads write language-agnostic output vectors?

**Method**: Patch head outputs from EN clean run into FR/YO corrupted run. Compatibility = (patched − ablated) / (clean − ablated). Floor control: same-language different-template patching ($\approx 1.0$ everywhere, validating instrument).

| Model | FR cross-compat | YO cross-compat |
|-------|----------------|----------------|
| 2B | 1.14 | 0.87 |
| 9B | 1.22 | 1.38 |
| 9B-it | 1.16 | 1.35 |
| 27B-it | 0.98 | 1.00 |

Cross-language compatibility $\approx 1.0$ or above across all models. EN head outputs are perfectly substitutable for FR/YO head outputs.

- **Verdict**: **Cleanest positive result for sharing in the entire project.** IOI heads write in a language-agnostic format. No dissociation with scale.

### 1.1.3 Representation vs. Computation Dissociation (Exp 17)

`Exp17_Rep_Space_Divergence.ipynb` — Gemma-2 9B, EN/FR/DE/HI.

Corrected cosine divergence (IOI-specific minus baseline):

| Language | Corrected divergence |
|----------|---------------------|
| FR | −0.055 (more similar than floor) |
| DE | +0.001 (no divergence) |
| HI | +0.290 (**real divergence**) |

- **Verdict**: Representations diverge for distant languages (HI) even on shared-computation tasks (Exp 14 showed computation is shared). **Representation and computation are dissociable** — a model can share circuits while its internal representations encode language-specific geometry. This is consistent with $\varepsilon > 0$ for distant languages but does NOT imply circuit separation.

---

## 1.2 Agreement Task Sharing (Exp 15, 15b)

### 1.2.1 Agreement Baselines (Exp 15)

`Exp15_Agreement_Scale.ipynb` — Gemma-2 2B, 27B-it. EN/FR.

Baselines only (head ranking and patching cells commented out, never executed):
- 2B: EN acc = 0.967, FR\_matched = 0.950
- 27B-it: EN acc = 0.983, FR\_matched = 0.800

- **Verdict**: *Incomplete* — no circuit-sharing results.

The drop in FR\_matched accuracy at 27B (0.950 → 0.800) is suggestive of degradation but without circuit-level analysis, this is a task-level observation only.

### 1.2.2 Multi-Language Agreement (Exp 15b)

`Exp15b_Agreement_MultiLang.ipynb` — Gemma-2 2B/9B/9B-it/27B-it. FR/DE/HI/AR/SW.

**Gate failures and exclusions**: AR entirely dropped — all three verb pairs share the same first token between singular and plural forms (tokenizer collision), making the task structurally unmeasurable. This is more severe than a gate failure: no AR data exists. SW gate failure across all models (matched acc 0.18–0.38). HI gate failure at 9B base (1 of 3 seeds), 9B-it (1 of 3), and all seeds at 27B-it.

Cross-language head-output compatibility (where gates passed, floor ≈ 1.0):

| Model | FR compat | FR floor | DE compat | DE floor | ret EN→lang (FR) |
|-------|----------|---------|----------|---------|-----------------|
| 2B | +1.27 | +0.99 | +0.84 | +0.99 | +0.01 |
| 9B | +1.92 | +0.99 | +1.62 | +0.99 | +0.12 |
| 9B-it | +1.67 | +1.03 | +1.39 | +1.03 | +0.02 |
| 27B-it | +4.32 | +1.02 | +1.60 | +1.02 | +0.77 |

**Interpreting compat > 1.** The formula is `compat = (patched − ablated) / (clean − ablated)`. The denominator (`gap = clean − ablated`) measures how much ablating EN-ranked top-10 heads hurts the target-language task. When ablation barely hurts (gap is small), any patching signal is amplified into a large ratio. At 27B FR, `retained EN-heads→lang = +0.77` means those heads account for only 23% of FR's agreement logit-diff — the denominator is small, inflating compat to 4.32. This does NOT mean EN heads help FR 4× more than its own heads; it means the instrument's resolution degrades when the selected heads are less relevant to the target language. A compat of 4.32 should be read as "sharing is present" (patched > ablated), not as "sharing is 4× stronger than self."

The notebook's own reading guide (cell 8 markdown) states the theory prediction:
> "Under the epsilon prediction: IOI compat ~ 1 everywhere; agreement compat < 1, lower for HI/AR/SW than FR/DE, lower at larger scale."

No separate `cost_derivation.md` or theory-derivation document exists in the repository. This notebook markdown is the sole statement of the theory's prediction for this experiment.

**Actual results vs. prediction**: Agreement compat > 1 (not < 1), increasing with scale (not decreasing), and for close languages only (HI/AR/SW lost to instrument failures). Every directional prediction is violated on the measurable cells. No conclusion/summary cell exists in the notebook beyond the raw output table.

- **Verdict**: **Contradicts the ε/N\* prediction** on every measurable dimension. But the measurable cells are limited to close languages (FR, DE) — HI, AR, and SW are lost to tokenizer collisions or accuracy gate failures. The compat >> 1 values at large scale are partly a denominator artifact that inflates the apparent magnitude of sharing while still confirming its direction.

---

## 1.3 Depth Patching (Exp 16)

`Exp16_v3_Residual_Stream_Patching.ipynb` — Gemma-2 2B/9B/27B, EN/FR/DE/HI.

Residual-stream patching at each layer. Cross-scale summary:
- 2B FR: PARTIAL (24/26 layers floor-ok)
- DE: PARTIAL (18/26)
- HI: Early split at layer 0 — confounded by script difference (Devanagari vs. Latin)
- Larger models: also PARTIAL verdicts

- **Verdict**: *Unresolved* for FR/DE (no clean split depth identified). HI result confounded by script. The depth-patching instrument doesn't cleanly resolve the sharing-depth question for close languages.

**Earlier versions** (`Exp16_Residual_Stream_Patching.ipynb`, `Exp16_v2_...`): superseded by v3.

---

## 1.4 BLOOM Experiments (Exp 18–23)

### 1.4.1 Preflight Gates (Exp 18–20)

`Exp18_Preflight_BLOOM.ipynb`: BLOOM 1b7/7b1, 7 languages (EN/FR/ES/DE/ZH/AR/HI). IOI gates pass for all languages on both models. Tokenization fertility measured.

`Exp19_Preflight_Tasks.ipynb`: Mixed pass/fail for negation, antonym, agreement, gender, classifier. Agreement accuracy for AR on bloom-1b7: 0.604 (borderline).

`Exp20_Preflight_Round2.ipynb`: Word order, translation, ZH classifier, AR numeral polarity. **Critical issue**: word order has `first_token_collision = 1.00` for some language pairs — the tokenizer produces identical first tokens for distinct word orders, making the task structurally unresolvable by next-token prediction.

### 1.4.2 Computation vs. Representation (Exp 21)

`Exp21_Computation_vs_Representation.ipynb` — BLOOM 7b1, 7 languages.

Two axes measured: computation sharing ($C_\text{resid}$, $C_\text{attn}$, $C_\text{mlp}$ from patching) and representation sharing ($R_\text{probe}$, $R_\text{cos}$, $R_\text{lang}$ from probes).

IOI: $C_\text{resid} \approx 0.98$ for EN→AR on bloom-1b7 — **computation is shared even across very distant language pairs**.

- **Verdict**: IOI sharing replicates on BLOOM, consistent with Gemma results.

### 1.4.3 Translation vs. Word Order (Exp 22)

`Exp22_Translation_WordOrder_Patching.ipynb` — BLOOM 7b1.

- Translation: $C_\text{resid}$ = 0.84–1.13 → **SHARED**
- Word order: frac\_flip = 0.03–0.19 → **NOT SHARED**

- **Verdict**: **First clean example of language-specific computation.** Translation shares circuits; word order does not. This is the strongest empirical distinction in the natural-data experiments.

### 1.4.4 Representation Alignment (Exp 23)

`Exp23_Representation_Wu_Lim.ipynb` — BLOOM 7b1. Semantic Hub and Lim-style analysis. Representation measures (CKA, hub\_rel\_mid) largely align with patching results — high similarity for IOI/translation, lower for word order.

- **Verdict**: Convergent evidence with patching experiments.

### 1.4.5 Complete Task × Language × Model Audit (Exp 18–22)

Every cell attempted across the five BLOOM notebooks, with quoted gate outputs and sharing measurements. Gate criteria (Exp 18–20): accuracy ≥ 0.75, Wilson 95% CI lower bound > 0.5, ≥ 90% items scorable, ≥ 8 single-token names (IOI only). A cell must pass the gate in **both** models to proceed to sharing measurement. Computation-sharing ($C$) uses floor-corrected cross-language patching compat (Exp 21–22); representation-sharing ($R$) uses label-direction transfer or retrieval (Exp 21–22). `NaN` = floor or ceiling too low to interpret.

---

#### A. IOI (Indirect Object Identification)

**Gate (Exp 18, Cell 6):**

| model | lang | name\_set | n\_names | acc | gate | reason if fail |
|---|---|---|---|---|---|---|
| bloom-1b7 | EN | native | 12 | 1.000 | **PASS** | |
| bloom-1b7 | FR | native | 11 | 1.000 | **PASS** | |
| bloom-1b7 | ES | native | 12 | 1.000 | **PASS** | |
| bloom-1b7 | DE | native | 12 | 0.930 | **PASS** | |
| bloom-1b7 | ZH | native | 0 | NaN | **FAIL** | 0 single-token native names |
| bloom-1b7 | ZH | en\_names | 12 | 1.000 | **PASS** | fallback to EN names |
| bloom-1b7 | AR | native | 12 | 1.000 | **PASS** | |
| bloom-1b7 | HI | native | 10 | 0.955 | **PASS** | |
| bloom-7b1 | EN | native | 12 | 1.000 | **PASS** | |
| bloom-7b1 | FR | native | 11 | 1.000 | **PASS** | |
| bloom-7b1 | ES | native | 12 | 1.000 | **PASS** | |
| bloom-7b1 | DE | native | 12 | 0.910 | **PASS** | |
| bloom-7b1 | ZH | native | 0 | NaN | **FAIL** | 0 single-token native names |
| bloom-7b1 | ZH | en\_names | 12 | 1.000 | **PASS** | fallback to EN names |
| bloom-7b1 | AR | native | 12 | 1.000 | **PASS** | |
| bloom-7b1 | HI | native | 10 | 0.995 | **PASS** | |

All 7 languages pass (ZH via EN-name fallback). Agreement-form collision rates (Exp 18): DE 0.4, HI 0.6, EN/FR/ES/AR 0.0, ZH no agreement forms.

**Sharing (Exp 21, Cell 5 + Exp 22, Cell 6 R\_retr):** All cells use EN names. R\_probe / R\_cos are NaN for IOI because the label (which name is the indirect object) has no fixed direction; R\_retr (label-free retrieval) is used instead.

| model | lang | n | clean\_acc | C\_resid | C\_attn | C\_mlp | R\_retr | R\_lang | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 24 | 1.00 | 1.09 | 0.91 | NaN | 0.78 | 0.87 | **SHARED** |
| bloom-1b7 | FR | 24 | 1.00 | 0.91 | NaN | NaN | 0.78 | 0.87 | **SHARED** |
| bloom-1b7 | ES | 24 | 1.00 | 0.84 | 0.62 | NaN | 0.96 | 0.88 | **SHARED** |
| bloom-1b7 | DE | 24 | 0.96 | 1.13 | 0.60 | NaN | 0.57 | 0.62 | **SHARED** |
| bloom-1b7 | ZH | 24 | 1.00 | 0.87 | 0.76 | NaN | 0.80 | 0.65 | **SHARED** |
| bloom-1b7 | AR | 24 | 1.00 | 0.98 | NaN | NaN | 0.70 | 0.83 | **SHARED** |
| bloom-1b7 | HI | 24 | 0.96 | 0.98 | 0.79 | NaN | 0.67 | 0.61 | **SHARED** |
| bloom-7b1 | EN | 24 | 1.00 | 1.07 | NaN | NaN | 1.00 | 0.94 | **SHARED** |
| bloom-7b1 | FR | 24 | 1.00 | 0.97 | NaN | NaN | 0.96 | 0.94 | **SHARED** |
| bloom-7b1 | ES | 24 | 1.00 | 0.97 | NaN | NaN | 1.00 | 0.94 | **SHARED** |
| bloom-7b1 | DE | 24 | 0.96 | 1.11 | NaN | NaN | 0.74 | 0.78 | **SHARED** |
| bloom-7b1 | ZH | 24 | 1.00 | 1.00 | NaN | NaN | 0.98 | 0.82 | **SHARED** |
| bloom-7b1 | AR | 24 | 1.00 | 1.08 | NaN | NaN | 0.89 | 0.91 | **SHARED** |
| bloom-7b1 | HI | 24 | 1.00 | 0.98 | NaN | NaN | 0.76 | 0.80 | **SHARED** |

IOI is uniformly **SHARED** across all 7 languages and both models. $C_\text{resid}$ ranges 0.84–1.13 (median ≈ 0.98). The task uses identical EN names in every language, so this is expected — the only thing that varies is the template language.

---

#### B. Negation

**Gate (Exp 19, Cell 5):**

| model | lang | n | acc | ci\_lo | acc\[kind=aff\] | acc\[kind=neg\] | gate | reason if fail |
|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 20 | 0.850 | 0.64 | 0.9 | 0.8 | **PASS** | |
| bloom-7b1 | EN | 20 | 1.000 | 0.84 | 1.0 | 1.0 | **PASS** | |
| bloom-1b7 | FR | 20 | 0.750 | 0.53 | 0.6 | 0.9 | **PASS** | |
| bloom-7b1 | FR | 20 | 0.900 | 0.70 | 0.8 | 1.0 | **PASS** | |
| bloom-1b7 | ES | 20 | 0.900 | 0.70 | 0.9 | 0.9 | **PASS** | |
| bloom-7b1 | ES | 20 | 0.950 | 0.76 | 0.9 | 1.0 | **PASS** | |
| bloom-1b7 | ZH | 20 | 0.750 | 0.53 | 0.7 | 0.8 | **PASS** | |
| bloom-7b1 | ZH | 20 | 0.800 | 0.58 | 0.8 | 0.8 | **PASS** | |
| bloom-1b7 | DE | 20 | 0.500 | 0.30 | 1.0 | 0.0 | **FAIL** | acc\[kind=neg\]=0.0: model copies affirmative |
| bloom-7b1 | DE | 20 | 0.500 | 0.30 | 1.0 | 0.0 | **FAIL** | acc\[kind=neg\]=0.0: model copies affirmative |
| bloom-1b7 | AR | 20 | 0.600 | 0.39 | 0.8 | 0.4 | **FAIL** | accuracy 0.600 < 0.75 |
| bloom-7b1 | AR | 20 | 0.600 | 0.39 | 0.8 | 0.4 | **FAIL** | accuracy 0.600 < 0.75 |
| bloom-1b7 | HI | 20 | 0.400 | 0.22 | 0.5 | 0.3 | **FAIL** | accuracy 0.400 < 0.75; negative mean margin |
| bloom-7b1 | HI | 20 | 0.500 | 0.30 | 0.7 | 0.3 | **FAIL** | accuracy 0.500 < 0.75 |

4 languages pass (EN, FR, ES, ZH). DE fails because both models score 0% on negated items (they copy the affirmative). AR fails on accuracy. HI fails with near-chance or below-chance performance.

**Sharing (Exp 21, Cell 5):** R\_probe / R\_cos are NaN for negation (no fixed label direction). R\_retr from Exp 22 used instead.

| model | lang | n | clean\_acc | C\_resid | C\_attn | C\_mlp | R\_retr | R\_lang | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 20 | 0.85 | 0.95 | NaN | NaN | 0.95 | 0.92 | **SHARED** |
| bloom-1b7 | FR | 20 | 0.75 | 0.87 | NaN | NaN | 0.97 | 0.92 | **SHARED** |
| bloom-1b7 | ES | 20 | 0.90 | 0.75 | 0.93 | 0.59 | 0.84 | 0.83 | **SHARED** |
| bloom-1b7 | ZH | 20 | 0.75 | 1.06 | NaN | NaN | 0.84 | 0.80 | **SHARED** |
| bloom-7b1 | EN | 20 | 1.00 | 0.90 | 0.92 | NaN | 1.00 | 0.96 | **SHARED** |
| bloom-7b1 | FR | 20 | 0.90 | 0.89 | 1.17 | 0.87 | 1.00 | 0.96 | **SHARED** |
| bloom-7b1 | ES | 20 | 0.95 | 0.81 | 1.05 | NaN | 0.92 | 0.91 | **SHARED** |
| bloom-7b1 | ZH | 20 | 0.80 | 0.99 | 1.14 | NaN | 0.79 | 0.89 | **SHARED** |

Negation is **SHARED** where measurable. $C_\text{resid}$ ranges 0.75–1.06. Note that FR and ZH clean accuracies are at or near the gate threshold (0.75–0.80), so the sharing measurement operates on a noisier signal.

---

#### C. Antonym

**Gate (Exp 19, Cell 5):**

| model | lang | n | scorable | acc | ci\_lo | gate | reason if fail |
|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-7b1 | EN | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-1b7 | FR | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-7b1 | FR | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-1b7 | ES | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-7b1 | ES | 6 | 1.00 | 1.000 | 0.61 | **PASS** | |
| bloom-1b7 | DE | 6 | 1.00 | 0.667 | 0.30 | **FAIL** | accuracy 0.667 < 0.75; ci\_lo 0.30 < 0.50 |
| bloom-7b1 | DE | 6 | 1.00 | 0.667 | 0.30 | **FAIL** | accuracy 0.667 < 0.75; ci\_lo 0.30 < 0.50 |
| bloom-1b7 | AR | 6 | 1.00 | 0.667 | 0.30 | **FAIL** | accuracy 0.667 < 0.75 |
| bloom-7b1 | AR | 6 | 1.00 | 0.500 | 0.19 | **FAIL** | accuracy 0.500 < 0.75 |
| bloom-1b7 | HI | 6 | 1.00 | 0.667 | 0.30 | **FAIL** | 1b7 fails (0.667 < 0.75); not both pass |
| bloom-7b1 | HI | 6 | 1.00 | 1.000 | 0.61 | **PASS** | but 1b7 fails, so cell excluded |
| bloom-1b7 | ZH | 6 | 0.83 | 1.000 | 0.57 | **FAIL** | scorable 0.83 < 0.90 (coverage gate) |
| bloom-7b1 | ZH | 6 | 0.83 | 1.000 | 0.57 | **FAIL** | scorable 0.83 < 0.90 (coverage gate) |

3 languages pass (EN, FR, ES). **No sharing measurement was designed or run for the antonym task.** Exp 21 states it covers "IOI, negation, agreement, gender" only. The antonym cells that passed remain behavioural-only.

---

#### D. Agreement (Subject–Verb Number)

**Gate (Exp 19, Cell 5):**

| model | lang | n | acc | ci\_lo | acc\[congr=F\] | acc\[congr=T\] | gate | reason if fail |
|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 48 | 0.958 | 0.86 | 0.917 | 1.000 | **PASS** | |
| bloom-7b1 | EN | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-1b7 | FR | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-7b1 | FR | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-1b7 | ES | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-7b1 | ES | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-1b7 | HI | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-7b1 | HI | 48 | 1.000 | 0.93 | 1.000 | 1.000 | **PASS** | |
| bloom-1b7 | DE | 48 | 0.583 | 0.44 | 0.583 | 0.583 | **FAIL** | accuracy 0.583 < 0.75; ci\_lo 0.44 |
| bloom-7b1 | DE | 48 | 0.750 | 0.61 | 0.708 | 0.792 | PASS alone | but 1b7 fails, so cell excluded |
| bloom-1b7 | AR | 48 | 0.604 | 0.46 | 0.542 | 0.667 | **FAIL** | accuracy 0.604 < 0.75 |
| bloom-7b1 | AR | 48 | 0.688 | 0.55 | 0.667 | 0.708 | **FAIL** | accuracy 0.688 < 0.75 |
| — | ZH | — | — | — | — | — | **N/A** | no agreement paradigm in Chinese |

4 languages pass (EN, FR, ES, HI). DE excluded because 1b7 fails (0.583). AR fails on both models.

**Sharing (Exp 21, Cell 5):** R\_probe and R\_cos are interpretable for agreement (the label direction = singular vs. plural). R\_retr from Exp 22 included.

| model | lang | n | clean\_acc | C\_resid | C\_attn | C\_mlp | R\_probe | R\_cos | R\_retr | R\_lang | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 24 | 0.92 | 1.05 | 0.90 | NaN | NaN | NaN | 0.37 | 0.87 | **SHARED** |
| bloom-1b7 | FR | 24 | 1.00 | 0.70 | 1.05 | NaN | 1.22 | 0.86 | 0.43 | 0.87 | **SHARED** |
| bloom-1b7 | ES | 24 | 1.00 | 0.65 | NaN | 0.49 | 1.20 | 0.76 | 0.52 | 0.88 | **PARTIALLY SHARED** |
| bloom-1b7 | HI | 24 | 1.00 | 0.50 | NaN | NaN | 0.92 | 0.32 | 0.09 | 0.64 | **WEAKLY SHARED** |
| bloom-7b1 | EN | 24 | 1.00 | 0.92 | NaN | NaN | NaN | 0.73 | 0.48 | 0.92 | **SHARED** |
| bloom-7b1 | FR | 24 | 1.00 | 0.81 | 0.86 | 0.45 | 2.00 | 0.59 | 0.52 | 0.92 | **SHARED** |
| bloom-7b1 | ES | 24 | 1.00 | 0.64 | NaN | NaN | 1.11 | 0.59 | 0.52 | 0.93 | **PARTIALLY SHARED** |
| bloom-7b1 | HI | 24 | 1.00 | 0.44 | NaN | NaN | 0.83 | 0.18 | 0.04 | 0.82 | **WEAKLY SHARED** |

Agreement shows a gradient: EN/FR are strongly shared ($C_\text{resid}$ 0.70–1.05), ES is partially shared (0.64–0.65), and HI is weakly shared ($C_\text{resid}$ 0.44–0.50, $R_\text{retr}$ near chance at 0.04–0.09). Hindi agreement morphology differs substantially from European patterns, and the low R\_retr confirms the representations are not aligned.

---

#### E. Gender (Grammatical)

**Gate (Exp 19, Cell 5):** Only languages with grammatical gender were tested.

| model | lang | n | acc | ci\_lo | gate | reason if N/A |
|---|---|---|---|---|---|---|
| bloom-1b7 | FR | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-7b1 | FR | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-1b7 | ES | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-7b1 | ES | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-1b7 | AR | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-7b1 | AR | 24 | 0.958 | 0.80 | **PASS** | |
| bloom-1b7 | HI | 24 | 1.000 | 0.86 | **PASS** | |
| bloom-7b1 | HI | 24 | 0.875 | 0.69 | **PASS** | |
| — | EN | — | — | — | **N/A** | no grammatical gender |
| — | DE | — | — | — | **N/A** | not tested (3-gender system complex) |
| — | ZH | — | — | — | **N/A** | no grammatical gender |

All 4 tested languages pass.

**Sharing (Exp 21, Cell 5):** Anchor is FR; FR itself uses ES as anchor. R\_probe and R\_cos are interpretable (label direction = masculine vs. feminine).

| model | lang | donor | n | clean\_acc | C\_resid | C\_attn | C\_mlp | R\_probe | R\_cos | R\_retr | R\_lang | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | FR | es | 24 | 1.00 | 0.77 | NaN | 0.50 | 0.50 | 0.47 | 0.22 | 0.89 | **PARTIALLY SHARED** |
| bloom-1b7 | ES | fr | 24 | 1.00 | 0.91 | NaN | 0.76 | 0.67 | 0.52 | 0.22 | 0.89 | **SHARED** |
| bloom-1b7 | AR | fr | 24 | 1.00 | 0.65 | 0.64 | 0.60 | 0.12 | 0.02 | 0.09 | 0.11 | **WEAKLY SHARED** |
| bloom-1b7 | HI | fr | 24 | 1.00 | −0.24 | NaN | 0.11 | −0.25 | −0.21 | 0.22 | 0.73 | **SEPARATED** |
| bloom-7b1 | FR | es | 24 | 1.00 | 0.79 | NaN | 0.58 | 0.42 | 0.47 | 0.22 | 0.93 | **PARTIALLY SHARED** |
| bloom-7b1 | ES | fr | 24 | 1.00 | 0.88 | NaN | 0.70 | 0.38 | 0.53 | 0.22 | 0.93 | **SHARED** |
| bloom-7b1 | AR | fr | 24 | 0.96 | 0.49 | 0.55 | 0.47 | 0.00 | 0.04 | 0.09 | 0.23 | **WEAKLY SHARED** |
| bloom-7b1 | HI | fr | 24 | 0.88 | 0.32 | NaN | 0.19 | −0.50 | −0.11 | 0.22 | 0.85 | **SEPARATED** |

Gender shows the strongest task-epsilon gradient. FR↔ES share computation ($C_\text{resid}$ 0.77–0.91) and partially share representations. AR is weakly shared ($C_\text{resid}$ 0.49–0.65, $R_\text{probe}$ near zero). HI is **separated**: negative $C_\text{resid}$ (−0.24 on 1b7) and negative $R_\text{probe}$ mean cross-language patching *hurts* — the gender circuits are language-specific.

---

#### F. ZH Classifier

**Gate — first attempt (Exp 19, Cell 5):**

| model | lang | n | scorable | acc | ci\_lo | gate | reason |
|---|---|---|---|---|---|---|---|
| bloom-1b7 | ZH | 16 | 0.75 | 0.833 | 0.55 | **FAIL** | scorable 0.75 < 0.90 (coverage gate) |
| bloom-7b1 | ZH | 16 | 0.75 | 0.833 | 0.55 | **FAIL** | scorable 0.75 < 0.90 (coverage gate) |

**Gate — fixed items (Exp 20, Cell 5):** Candidates changed to full noun-phrase form (`三{classifier}{noun}`) to eliminate the coverage issue.

| model | lang | n | scorable | acc | ci\_lo | gate |
|---|---|---|---|---|---|---|
| bloom-1b7 | ZH | 16 | 1.00 | 0.938 | 0.72 | **PASS** |
| bloom-7b1 | ZH | 16 | 1.00 | 1.000 | 0.81 | **PASS** |

Passes gate after redesign. First-token collision = 0.75 (i.e. 75% of items share their first token between correct and foil candidate). **No sharing measurement was run.** Exp 22 markdown: "ZH classifier and AR numeral polarity have no counterpart language, so there is no cross-language donor." The task remains a behavioural high-ε endpoint only.

---

#### G. AR Numeral Polarity

**Gate (Exp 20, Cell 5):**

| model | lang | n | scorable | acc | ci\_lo | acc\[cue=F\] | acc\[cue=T\] | gate |
|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | AR | 16 | 1.00 | 0.938 | 0.72 | 0.917 | 1.0 | **PASS** |
| bloom-7b1 | AR | 16 | 1.00 | 0.938 | 0.72 | 0.917 | 1.0 | **PASS** |

Passes gate. Zero first-token collision. **No sharing measurement was run** — same reason as classifier: no counterpart language exists for numeral-gender polarity reversal. Behavioural high-ε endpoint only.

---

#### H. Word Order

**Gate (Exp 20, Cell 5):** Scored by full-sentence log-probability, not first-token, so the high first-token collision rates (1.00 for EN/FR/ES/DE/HI; 0.67 for AR) do not affect the gate.

| model | lang | n | acc | ci\_lo | acc\[adj\] | acc\[adp\] | acc\[clause\] | gate |
|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-1b7 | FR | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-1b7 | ES | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-1b7 | DE | 18 | 0.833 | 0.61 | 0.667 | 1.0 | 0.833 | **PASS** |
| bloom-1b7 | ZH | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-1b7 | AR | 18 | 0.833 | 0.61 | 1.000 | 1.0 | 0.500 | **PASS** |
| bloom-1b7 | HI | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | EN | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | FR | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | ES | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | DE | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | ZH | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | AR | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |
| bloom-7b1 | HI | 18 | 1.000 | 0.82 | 1.000 | 1.0 | 1.000 | **PASS** |

All 7 languages pass. AR clause order is weakest (1b7: 0.500 on clause sub-phenomenon alone, but pooled accuracy still exceeds the gate).

**Sharing (Exp 22, Cell 6):** Word order uses `frac_flip` (fraction of a full order reversal from an opposite-order donor), **not** floor-corrected compat, because no within-language label flip exists. Values near 0 = not shared; near 1 = fully shared.

| model | lang | n | clean\_acc | frac\_flip (late median) | frac\_flip (max) | null (late median) | verdict |
|---|---|---|---|---|---|---|---|
| bloom-1b7 | EN | 18 | 1.00 | 0.05 | 0.05 | −0.04 | **NOT SHARED** |
| bloom-1b7 | FR | 18 | 1.00 | 0.06 | 0.07 | −0.07 | **NOT SHARED** |
| bloom-1b7 | ES | 18 | 1.00 | 0.08 | 0.09 | −0.11 | **NOT SHARED** |
| bloom-1b7 | DE | 18 | 0.83 | −0.18 | −0.17 | −0.01 | **NOT SHARED** |
| bloom-1b7 | ZH | 10 | 1.00 | 0.13 | 0.15 | 0.10 | **NOT SHARED** |
| bloom-1b7 | AR | 12 | 1.00 | 0.15 | 0.23 | −0.02 | **NOT SHARED** |
| bloom-1b7 | HI | 18 | 0.94 | 0.04 | 0.10 | 0.00 | **NOT SHARED** |
| bloom-7b1 | EN | 18 | 1.00 | 0.03 | 0.04 | −0.03 | **NOT SHARED** |
| bloom-7b1 | FR | 18 | 1.00 | 0.03 | 0.04 | −0.06 | **NOT SHARED** |
| bloom-7b1 | ES | 18 | 1.00 | 0.05 | 0.06 | −0.11 | **NOT SHARED** |
| bloom-7b1 | DE | 18 | 0.94 | −0.04 | 0.01 | 0.01 | **NOT SHARED** |
| bloom-7b1 | ZH | 10 | 1.00 | 0.19 | 0.22 | 0.11 | **NOT SHARED** |
| bloom-7b1 | AR | 12 | 1.00 | 0.19 | 0.23 | −0.03 | **NOT SHARED** |
| bloom-7b1 | HI | 18 | 1.00 | −0.01 | 0.01 | −0.01 | **NOT SHARED** |

Per sub-phenomenon (Exp 22, Cell 6, late-layer median frac\_flip):

| model | sub | EN | FR | ES | DE | ZH | AR | HI |
|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | adj | 0.13 | 0.17 | 0.25 | −0.30 | 0.11 | −0.03 | 0.08 |
| bloom-1b7 | adp | 0.01 | 0.07 | 0.05 | 0.03 | NaN | 0.35 | 0.07 |
| bloom-1b7 | clause | 0.05 | −0.03 | 0.01 | −0.62 | 0.08 | NaN | −0.07 |
| bloom-7b1 | adj | 0.11 | 0.14 | 0.21 | −0.31 | 0.01 | 0.03 | 0.01 |
| bloom-7b1 | adp | 0.02 | 0.05 | 0.04 | 0.13 | NaN | 0.33 | −0.07 |
| bloom-7b1 | clause | 0.01 | −0.07 | −0.04 | −0.06 | 0.16 | NaN | 0.04 |

Word order is uniformly **NOT SHARED**. frac\_flip is near zero everywhere (max 0.23 for AR/ZH). DE shows a negative flip (−0.18 pooled for 1b7), meaning the opposite-order donor pushes the model *further* toward the recipient's own order — anti-transfer. The adjective sub-phenomenon shows the largest positive values (up to 0.25 for ES), but even these are far from 1.0.

---

#### I. Translation

**Gate (Exp 20, Cell 5):**

| model | lang | n | acc | ci\_lo | gate |
|---|---|---|---|---|---|
| bloom-1b7 | FR | 150 | 0.987 | 0.95 | **PASS** |
| bloom-1b7 | ES | 150 | 0.987 | 0.95 | **PASS** |
| bloom-1b7 | DE | 150 | 0.800 | 0.73 | **PASS** |
| bloom-1b7 | ZH | 150 | 0.987 | 0.95 | **PASS** |
| bloom-1b7 | AR | 150 | 0.933 | 0.88 | **PASS** |
| bloom-1b7 | HI | 150 | 0.933 | 0.88 | **PASS** |
| bloom-7b1 | FR | 150 | 0.993 | 0.96 | **PASS** |
| bloom-7b1 | ES | 150 | 0.993 | 0.96 | **PASS** |
| bloom-7b1 | DE | 150 | 0.880 | 0.82 | **PASS** |
| bloom-7b1 | ZH | 150 | 0.993 | 0.96 | **PASS** |
| bloom-7b1 | AR | 150 | 0.967 | 0.92 | **PASS** |
| bloom-7b1 | HI | 150 | 0.967 | 0.92 | **PASS** |

All 6 non-EN languages pass. ZH translation has first-token collision = 0.49 (half the items share their first token), but this doesn't prevent scoring since full-sequence log-prob is used. DE is weakest (acc 0.800/0.880).

**Sharing (Exp 22, Cell 6):** Translation uses concept-patching (Dumas et al. design). R\_retr used for representation.

| model | lang | donor | n | clean\_acc | C\_resid | C\_attn | C\_mlp | R\_retr | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bloom-1b7 | FR | es | 24 | 1.00 | 0.92 | 1.17 | NaN | 1.0 | **SHARED** |
| bloom-1b7 | ES | fr | 24 | 1.00 | 0.89 | 0.93 | NaN | 1.0 | **SHARED** |
| bloom-1b7 | DE | fr | 24 | 0.79 | 1.13 | NaN | NaN | 1.0 | **SHARED** |
| bloom-1b7 | ZH | fr | 24 | 1.00 | 0.79 | NaN | NaN | 1.0 | **SHARED** |
| bloom-1b7 | AR | fr | 24 | 1.00 | 0.86 | NaN | NaN | 1.0 | **SHARED** |
| bloom-1b7 | HI | fr | 24 | 1.00 | 0.84 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | FR | es | 24 | 1.00 | 0.89 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | ES | fr | 24 | 1.00 | 0.83 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | DE | fr | 24 | 0.88 | 0.87 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | ZH | fr | 24 | 1.00 | 0.76 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | AR | fr | 24 | 1.00 | 0.80 | NaN | NaN | 1.0 | **SHARED** |
| bloom-7b1 | HI | fr | 24 | 1.00 | 0.77 | NaN | NaN | 1.0 | **SHARED** |

Translation is uniformly **SHARED**. $C_\text{resid}$ ranges 0.76–1.13 (median ≈ 0.85). $R_\text{retr}$ = 1.0 everywhere — perfect label-free nearest-neighbour retrieval — confirming that translation concepts occupy the same representational subspace across languages. DE 1b7 clean accuracy (0.79) is near threshold, but the sharing signal is still clear.

---

#### J. Factual Recall

**Not attempted.** None of the five BLOOM notebooks (Exp 18–22) include a factual-recall task.

---

#### Summary: Which tasks yielded usable verdicts?

Six tasks produced usable sharing/separation verdicts: IOI (7 languages, all SHARED), negation (4 languages, all SHARED), agreement (4 languages, gradient from SHARED for EN/FR to WEAKLY SHARED for HI), gender (4 languages, gradient from SHARED for ES to SEPARATED for HI), translation (6 languages, all SHARED), and word order (7 languages, all NOT SHARED). Four task-types never received a sharing/separation verdict: antonym (passed gate for EN/FR/ES but no sharing measurement was designed), ZH classifier and AR numeral polarity (both passed their Exp 20 gates but have no cross-language donor for patching), and factual recall (not attempted in any notebook).

---

## 1.5 Word Order and Gender: Direction Transfer (Exp 24–29)

### 1.5.1 Gender Feature Gate (Exp 24)

`Exp24_Concept_Gender_Feature_vs_Lookup.ipynb` — Gemma-2 2B/9B/9B-it/27B-it, FR/ES/AR/HI. 28 concepts, conflict counts: FR–ES 4/24 conflicting, FR–AR 17/11 conflicting.

Gate passed: gender concept is extractable in all models and languages.

### 1.5.2 Word Order CKA (Exp 25)

`Exp25_WordOrder_Parallel.ipynb` — BLOOM 1b7/7b1, 7 languages, parallel items (same scenes across languages).

Three sub-phenomena: clause order (SVO/SOV), adjective order, adposition (pre/post). Metrics: frac\_flip, CKA, R\_retr.

- **Clause order**: frac\_flip $\approx 0$ across SVO recipients (EN: −0.07, FR: 0.00, DE: −0.40). CKA high among European languages (0.95–1.00), very low European–HI/AR (0.13–0.18). Most cells: "separated computation / shared representation."
- **Adj order**: frac\_flip low (0.07–0.26 max). No strong transfer.
- **Adposition**: AR→HI 0.38 (only positive transfer cell above threshold).

- **Verdict**: Word-order computation is language-specific even when representations are shared (same-script languages). Confirms Exp 22 on a different model family with parallel items.

### 1.5.3 Word Order DTI (Exp 26, 27)

`Exp26_WordOrder_Calibrated.ipynb`: Three instrument problems identified; superseded by Exp 27.

`Exp27_A_WordOrder_Gate.ipynb` — Gemma-2 9B, 224 units (clause/adj/de\_frames).

Self-patch, overwrite, state-change checks pass. Same-language ceiling: V2←SUB flip = 0.738. Cross-language results:

| Category | Count |
|----------|-------|
| TRANSFER | 1 |
| WALL | 10 |
| AMBIGUOUS | 4 |

- **Verdict**: **INVERSION\_ABANDONED** — the DTI instrument fundamentally cannot resolve word-order transfer because ceiling controls invert in 10/15 cases. The single anomalous TRANSFER cell prevents claiming a clean word-order wall. This is an instrument failure, not a theory test.

### 1.5.4 Gender DTI (Exp 28)

`Exp28_B_Gender_DTI.ipynb` — Gemma-2 9B, 64 units, 6 cross-family concepts.

Same-language ceiling: AR ceil\_flip = 0.373. Three languages pass the accuracy gate (FR, ES, AR — HI fails at 0.62). AR passes both the accuracy and ceiling gates but is classified UNUSABLE by the DTI cross-language coherence check. Final tally: only 2 usable recipient languages (need $\geq 3$). ES = AMBIGUOUS, FR = AMBIGUOUS.

- **Verdict**: **INSTRUMENT\_INADEQUATE** — too few viable recipient languages.

### 1.5.5 DAS Direction Transfer (Exp 29)

`Exp29_DirectionTransfer_DAS_v5.ipynb` — **Gemma-2-9B-it** (instruction-tuned; confirmed from cell 3 model load). The most mature natural-data experiment (v5 after four iterations of instrument refinement). No base-model or 27B run exists in any Exp 29 notebook version.

**Method**: DAS-lite direction extraction, necessity + sufficiency gates. Pre-registered (cell 2: `prereg frozen at exp29_results_run4/prereg.json`). Self-tests pass.

**Gender** (v5, cell 20):
- FR→ES: TRANSFER ($R$ = 1.06)
- FR→HI: TRANSFER ($R$ = 0.56)
- FR→AR: TRANSFER ($R$ = 0.94)
- ES→HI: AMBIGUOUS ($R$ = 0.35) — the only non-TRANSFER cell
- 11/12 gender cells = TRANSFER

**Word order** (v5, cell 20):
- DE→EN: $R$ = 0.43 (AMBIGUOUS — margin shift is large but insufficient flips)
- DE→FR/ES/HI/JA/KO/TR: all WALL ($R$ = 0.00–0.11)
- 6/7 order cells = WALL

**Specificity**: Gender direction does not flip word order; word-order direction does not flip gender. 2×2 specificity confirmed.

- **Verdict**: **Strongest evidence for the task-dependent asymmetry.** Gender is shared (additive, necessary, specific, transfers across language families). Word order is language-specific (directions transfer within language but not across). Qualifications: (a) DE→EN order being AMBIGUOUS prevents a perfectly clean INVERSION\_SUPPORTED verdict; (b) tested on 9B-it only — whether results hold on base models or at other scales is untested.

**Earlier versions** (`Exp29_DirectionTransfer_DAS.ipynb` v1–v4): Progressive instrument refinement. v1 self-tests passed but cross-language run skipped (item banks not populated). v2–v4 added gating, bootstrap CIs, and sufficiency checks.

---

# PART 2: Synthetic Setups

These experiments test the cost-theoretic framework on toy models where $\tau$, $\varepsilon$, $K$, and $N$ can be controlled directly.

---

## 2.1 Capacity Competition — Theory Only (Exp 1)

`Experiment_1_Capacity_Competition.ipynb` (two identical copies)

**Setup**: Pure closed-form computation. No neural model. Tests whether the $N^*$ formula produces the predicted specialisation cascade under heterogeneous $(\tau, \varepsilon)$.

**Key results**:
- **Symmetric case** ($\tau_i = \tau, \varepsilon_i = \varepsilon$ for all $i$): Allocation jumps from 0 private to $K$ private in a single step — no smooth cascade. Heterogeneity is **required** for gradual peel-off.
- **Heterogeneous** $(\tau, \varepsilon)$: Distinct specialisation thresholds. Language 7 ($\tau = 0.40$) specialises at $N = 203$; language 1 ($\tau = 0.10$) at $N = 2073$.
- **Comparative statics**: Both confirmed — $\tau \uparrow \Rightarrow N^* \downarrow$ and $\varepsilon \uparrow \Rightarrow N^* \downarrow$.
- $\varepsilon$ effect quantified: $\varepsilon = 0.001 \Rightarrow N^* = 1551$; $\varepsilon = 1.0 \Rightarrow N^* = 1.1$.
- Joint phase diagram ($\tau \times \varepsilon \times N$) computed.

- **Verdict**: Theory self-consistent. The formula does what it claims. The real question is whether neural networks obey these closed-form predictions.

---

## 2.2 Learnable Allocation — Neural Model (Exp 2)

### 2.2.1 Main Result (`Experiment_2_Full.ipynb`)

The definitive Exp 2 version. Parity task, $K$ languages, soft-gated shared/private routes.

**Three bugs found and fixed during development**:
1. Soft-gate collapse (gate saturated to one route regardless of cost)
2. $\sigma \neq \varepsilon$ calibration error (injected noise did not produce intended representational penalty)
3. Budget-flooring artifact (minimum allocation masked genuine separation)

**After fixes**: SGD peel-off ORDER matches theory exactly — languages specialise in descending $(\tau, \sigma)$ order. Quantitative timing gap $\sim 2\times$ between predicted and observed $N^*$.

$\varepsilon = 0$ sanity check: cleanly confirms sharing wins when no floor exists.

- **Verdict**: Theory ORDER confirmed in neural setting. Quantitative calibration has a $2\times$ gap (not surprising for a mean-field theory vs. SGD dynamics).

### 2.2.2 Calibrated Version (`Experiment_2_Revised_Calibrated_Shared_Private_fixed.ipynb`)

Parity task, $K = 4$, controlled $\varepsilon$ (bit erasure) and $\tau$ (shortcut cue $q$).

- $\varepsilon_\text{Bayes}$: 0.185 (low erase) or 0.590 (high erase)
- $\tau_\text{proxy}$: 0.0 ($q = 0.5$) or 0.81 ($q = 0.95$)

Calibration showed shared-route loss floors land correctly. **But**: shortcut calibration gave NaN for $\tau_\text{capacity\_proxy}$ (low-$\tau$ condition never reached target loss = 0.25 at any width tested).

**CRITICAL**: The full allocation sweep was **never run** (`RUN_FULL = False`, code commented out). The notebook establishes the calibration framework but the decisive test was never executed.

- **Verdict**: *Design complete, experiment incomplete.* Reason: not documented (likely compute or debugging time).

### 2.2.3 Anchor Bias (`Experiment_2b_Anchor_Bias.ipynb`)

**Question**: Can imbalanced training alone (no injected noise) produce an $\varepsilon$-like gap?

4 languages with different encoders. Language 0 = anchor (70% sampling).

- Private floors: $\sim 0.015$–$0.027$ per language (generic finite-width cost)
- Biased vs. uniform: clear gap under biased training (anchor $\sim 0.054$, minorities $\sim 0.12$–$0.16$ at width 450). Under uniform sampling, all languages track closely.
- Extended to width 4800 (GPU): ratio stays $\sim 2.1$–$3.1\times$, absolute gap slowly shrinks then stabilises
  - Width 1600: gap = [0.054, 0.072, 0.087]
  - Width 4800: gap = [0.041, 0.059, 0.078]
- Cohort-invariance: lang1 loss roughly stable across companion swaps ($\sim 0.11$–$0.14$)

- **Verdict**: *Suggestive, not conclusive.* Gap stabilisation between widths 900–4800 is consistent with genuine $\varepsilon$ but the absolute gap continues to shrink slowly — could still be converging to zero on a longer timescale. Not strong enough to establish A2 ($\varepsilon$ does not shrink with capacity).

**Earlier/intermediate versions** (`Experiment_2_Learnable_Allocation.ipynb` + 1 copy, `Experiment_2_Learnable_Shared_Private_Allocation.ipynb`, `Experiment_2_Revised_Calibrated_Shared_Private.ipynb`): Superseded. The `_Revised_` version without `_fixed` suffix has a code bug (tuple unpacking mismatch) and no results.

---

## 2.3 PCFG Number Agreement (Exp 3)

`Experiment_3_Number_Agreement_PCFG.ipynb` + 5 related notebooks (`Experiment_3_Diagnostic_Standalone.ipynb`, `Experiment_3_Weight4_Extended_Standalone.ipynb`, `PCFG_git.ipynb`, `v8_multilingual_pcfg.ipynb`, `v9_multilingual_pcfg.ipynb`)

**Setup**: PCFG-generated number-agreement task. $\tau$ manipulated via attachment distance; $\varepsilon$ manipulated via lookup-table divergence between languages.

**$\tau$ twin sweep (A vs. B)**:
- A ($\tau_\text{intervene} = 0.05$): acc = 0.999
- B ($\tau_\text{intervene} = 0.95$): acc = 0.958
- $\tau$ manipulation works — genuine separation in the capacity curve.

**$\varepsilon$ twin sweep (A vs. C)**:
- eps\_cost at cap = 4: −0.001; cap = 6: +0.063; cap = 8: +0.070; cap = 12: +0.004
- 5-seed replication: eps\_cost = 0.044 at cap = 4, 0.052 at cap = 6

The $\varepsilon$ mechanism **partially engages**: positive eps\_cost at cap = 6–8 means the joint model pays MORE for the high-$\varepsilon$ condition than the low-$\varepsilon$ condition, in the predicted direction. But this cost does NOT translate into differential $N^*$ — both conditions solve the task at the same capacity threshold.

Cross-language eval: anchor-trained on A, eval on C gave 0.74 (large gap, expected).

**$\varepsilon$ sweep across all PCFG designs**: Across 4 different designs (4-cell table, 256-cell table, twin-pair cascades, full 8-language sweeps), $\varepsilon$-as-lookup-table-divergence never produces differential $N^*$.

**Three failure modes identified**:
1. Override absorbed by fixed-width MLP (model learns to ignore the rare-event manipulation)
2. Model learns majority-class heuristic to bypass rare-event override
3. Task hits accuracy ceiling before capacity competition can emerge

- **Verdict**: $\tau$ manipulation works. **$\varepsilon$ mechanism partially engages** (positive eps\_cost at cap = 6–8) **but does not produce differential $N^*$** — the neural network circumvents it before the cost difference can shift the allocation threshold. This is a design flaw in the $\varepsilon$ operationalisation, not necessarily a theory falsification, but it means the PCFG setup cannot test the $N^*$ prediction.

---

## 2.4 Conflicting Gender (Exp 4)

`Experiment4_ConflictingGender_Followups.ipynb` (+ 1 identical copy)

**Setup**: Shared-vocabulary conflicting gender. 8 nouns, overlap parameter controls conflict severity.

Dense grid results:

| overlap | n\_embd = 2 | n\_embd = 4 | n\_embd = 6 |
|---------|------------|------------|------------|
| 0.0 | 0.462 | — | 1.000 |
| 0.5 | — | — | solved |
| 1.0 | — | solved | solved |

**Key finding**: $N^*$ behaves as a **binary switch**, not a gradient. overlap = 0.5 and overlap = 0.0 produce essentially the same threshold ($N^* \approx 5$). The disjoint-vocabulary control that would distinguish conflict cost from memorisation cost was set up but **never run** (sweep interrupted partway). No disjoint-vocabulary results exist in the notebook's saved outputs.

- **Verdict**: *Design flaw* — the overlap parameter does not produce graded $N^*$ as predicted. Binary-switch behaviour suggests the model solves the conflict/no-conflict distinction categorically rather than through continuous cost-sharing tradeoffs.

---

## 2.5 Allen-Zhu Paired CFGs (Exp 5)

`exp5_allenzhu_paired_cfgs.ipynb`

**Setup**: Allen-Zhu-style depth-7 layered CFGs with 3 terminals. GPT-2 with RoPE (12 layers, 12 heads). 4 conditions: mono\_a, mono\_b, bi\_shared, bi\_disjoint. CYK parser validation of generated strings.

**Pre-checks**:
- Rule overlap between grammars: 0.208 (pass threshold $< 0.25$)
- N-gram CE: 1.58 → 1.05 bits from order 0 to 6 (genuine long-range structure)
- Pre-check 1 (mono-A, d\_model = 768): loss converged at $\sim 0.745$ after 20K steps

**Full sweep**: 4 conditions × 5 capacities × 3 seeds = 60 runs. Set up but **not completed** — estimated compute: $\sim$100–150 A100 GPU-hours.

- **Verdict**: *Best-designed $\varepsilon$ experiment in the repo* (proper controls, CYK validation, budget-matched conditions), but **never completed** due to compute requirements. No $\varepsilon$ or $N^*$ analysis results exist.

---

## 2.6 Shared-Tree / Divergent Linearisation (Exp 6)

### 2.6.1 Main Result (`Experiment6_SharedTree_DivergentLinearization_CLEANED (1).ipynb`)

**THE KEY SYNTHETIC RESULT.** Same abstract tree, three linearisations:
- $L_A$: identity (left-to-right)
- $L_B$: partial reversal at levels 4–5
- $L_C$: full reversal

Gate checks: zero within-language string collisions; n-gram entropy parity across languages.

**Two configurations were run at d=96 with conflicting results.**

#### Configuration 1: v3 sweep (n\_head=12, n\_layer=12)

Cell 31 output (early-stopping convergence, NOT step-matched):

| $d$ | $\varepsilon_B$ | $\varepsilon_C$ |
|-----|----------------|----------------|
| 24 | +0.037 | +0.060 |
| 48 | +0.033 | +0.050 |
| 96 | **+0.025** | **+0.047** |
| 192 | −0.033 | +0.232 |
| 384 | +0.198 | +0.302 |
| 768 | −0.033 | +0.014 |

Results at d ≥ 192 are clearly unreliable (L\_A eps should be ≈ 0 but reads +0.059 at d=192, +0.214 at d=384). At d ≤ 96, $\varepsilon_C > \varepsilon_B > 0$ — ordering matches linearisation distance.

#### Configuration 2: big-pool h4 (n\_head=4, ConvergenceCheck notebook)

`Experiment6_ConvergenceCheck_d96.ipynb` — 3 seeds, 90K steps, extended to 180K.

Cell 12 output:
```
entropy floor H_sh = 0.7545
seed | CE-gap B @90K  CE-gap B @180K
   0 |       -0.0085        -0.0065
   7 |       -0.0092        -0.0071
```

CE-gap B ≈ −0.009, flat through 180K steps. Cell 13 concludes: "the 'no ε floor at d=96' conclusion holds."

**Important caveat**: this notebook only checks CE-gap B (not per-language eps\_pure for all three languages) and uses n\_head=4, a different architecture from the v3 sweep's n\_head=12.

#### Step-matched diagnostic (n\_head=12, main notebook cell 47)

Cell 47 matches step counts (both at 89,108 steps), then extends both by +30K in lockstep (to 119,108 each):

**At matched steps (89K each):**

| lang | CE\_shared | CE\_disjoint | eps\_pure |
|------|-----------|-------------|----------|
| L\_A | 0.7607 | 0.7659 | **−0.005** |
| L\_B | 0.7552 | 0.7168 | **+0.038** |
| L\_C | 0.7654 | 0.7152 | **+0.050** |

**After +30K lockstep extension (119K each):**

| lang | CE\_shared | CE\_disjoint | eps\_pure |
|------|-----------|-------------|----------|
| L\_A | 0.7541 | 0.7585 | **−0.005** |
| L\_B | 0.7441 | 0.7214 | **+0.023** |
| L\_C | 0.7671 | 0.7068 | **+0.060** |

#### Second seed (seed 7, cell 49, n\_head=12, fixed 90K steps)

| lang | CE\_shared | CE\_disjoint | eps\_pure |
|------|-----------|-------------|----------|
| L\_A | 0.7563 | 0.7660 | **−0.010** |
| L\_B | 0.7569 | 0.7111 | **+0.046** |
| L\_C | 0.7693 | 0.7050 | **+0.064** |

Cross-seed agreement:
```
L_A: seed 1 = −0.005, seed 2 = −0.010  (both near zero — correct, L_A is identity)
L_B: seed 1 = +0.023, seed 2 = +0.046  (both positive)
L_C: seed 1 = +0.060, seed 2 = +0.064  (both positive, C > B in both seeds)
```

#### The h4/h12 discrepancy

The two configurations disagree at d=96:
- **h4 (n\_head=4)**: eps ≈ 0 or slightly negative → "no epsilon floor"
- **h12 (n\_head=12)**: eps clearly positive, reproducible across 2 seeds → "epsilon persists"

Both are in the repository. The discrepancy is not resolved in any notebook. Possible explanations: (a) 4 heads vs 12 heads changes the attention-routing dynamics enough to alter whether linearisation cost persists; (b) convergence-matching protocol differences between the two setups.

Representational probes built: structural probe (rank 32), Procrustes, SVCCA, NT-identity probe. Own-lang probes showed L\_B/L\_C representational capacity worse than L\_A at small $d$, converging at large $d$.

Core diagnosis (cell 115): "$\varepsilon$ has nowhere to live in this design — all three languages share identical tokens and embeddings, so the shared space has no upstream bias to cost anything."

- **Verdict**: Co-training cost ordered by word-order distance at small $d$ (8–32), confirmed across configurations. At $d = 96$: **configuration-dependent** — h12 shows persistent positive $\varepsilon$ (2 seeds), h4 shows $\varepsilon \approx 0$ (2 seeds). A2 ($\varepsilon$ does not shrink with capacity) is violated under h4 but not clearly violated under h12. The conflict is unresolved.

### 2.6.2 Conversion Cost vs. $\varepsilon$ (`Exp6_conversion_cost_vs_eps.ipynb`)

**Question**: Does translator cost $a(g)$ track the $\varepsilon$ ordering?

d = 96, 2 seeds visible in saved output (seeds 0 and 1), 30K steps. $a(g)$ = AUC(translator\_$g$) − AUC(identity):
- Seed 0: $a(L_B)$ = 0.045, $a(L_C)$ = −0.000
- Seed 1: $a(L_B)$ = 0.063, $a(L_C)$ = −0.009

**CRITICAL**: $a(L_C) < a(L_B)$ — **opposite** of the $\varepsilon$ ordering ($\varepsilon_C > \varepsilon_B$ from Exp 6 main). Full reversal (L\_C) is trivially cheap to convert (just reverse the sequence) while partial reversal (L\_B) requires position-dependent logic.

Cell 14 markdown: "SPEED-ONLY PASS: at d=96 both translators may reach ~0 CE"

**Side-branch** (`Exp8_relabel_circuit_construction.ipynb`): Hand-built unembedding permutation achieves $\sim$100% exact match on relabel\_new → the $\sim 28\times$ $a(g)$ gap for relabel\_new is a **learning-dynamics artifact**, not a circuit-cost fact. This undermines $a(g)$ as a proxy for computational cost.

- **Verdict**: **Fail.** Conversion cost does not track $\varepsilon$, and may not measure circuit cost at all.

### 2.6.3 Transfer Test (`Exp6_transfer_test.ipynb`)

Prediction (frozen at cell 11, "Sun Nov 1 19:04:01 2026"): $S_C > S_B$ (L\_C should adapt faster since $a(L_C) < a(L_B)$).

Results per seed ($S_g$ = AUC(scratch) − AUC(finetune)):
- $S_B$: 0.033, 0.035, 0.023
- $S_C$: 0.022, 0.020, 0.019
- $S_C - S_B$ per seed: −0.011, −0.015, −0.004. Mean −0.010 ± 0.003.
- L\_B adapted faster than L\_C in **all 3 seeds**, opposite of prediction.
- Zero-shot CE: L\_B = 1.64, L\_C = 3.08 (L\_C much worse at zero-shot, as expected for full reversal).

- **Verdict**: **Fail** — the notebook itself labels this "FAIL — the reverse ordering held." The frozen prediction from the conversion-cost framework does not hold.

### 2.6.4 Syncretism / Budget-Matched (`Experiment6b_Syncretism_BudgetMatched_CLEANED.ipynb`)

Syncretism (terminal symbol collapse) as controlled $\varepsilon$. Budget-matched dedicated models as fair comparison.

Oracle check revealed: syncretism is partially resolvable from local context ($K = 32$ context window: ambiguous fraction only $\sim 0.31$).

Terminal-recovery probe, CE gap, interchange intervention all measured.

- **Verdict**: *Abandoned* — terminal collapse tests input destruction, not anchor-bias. The mechanism doesn't map cleanly onto the $\varepsilon$ concept.

### 2.6.5 S5-Relay Validation (`Experiment6b_v2_S5Relay_ValidationLadder.ipynb`)

Attempted S5-relay as harder substrate. Abandoned at validation ladder — S5 relay too hard for architecture (cliff at $m \approx 10$–$14$).

- **Verdict**: *Abandoned* — architecture limitation.

---

## 2.7 Dual-Role $\varepsilon$ Test (Exp 7)

### 2.7.1 Dual-Role (`Experiment7_DualRole_tAgree_CLEANED.ipynb` + 4 earlier variants)

**Setup**: Terminal $t_2$ has dual roles (ROLE-X / ROLE-Y); $L_A$: P(ROLE-X) = 0.903; $L_B$/$L_C$: P(ROLE-X) = 0.096. Budget-matched dedicated baselines ($d' = 8/16/56$ for $d = 8/16/32/96$). A' calibration control (anchor-identical second language). Convergence-matched training (90K steps).

**Multiple eval-construction bugs were found and fixed** (divide-by-zero default masquerading as 0.000, per-language reweighting bug, role-probe on wrong distribution). Each initially looked like a finding. After fixes:

| $d$ | lang | shared minacc | dedicated minacc | gap |
|-----|------|--------------|-----------------|-----|
| 8 | L\_B | 0.980 | 0.000 | **+0.980** |
| 16 | L\_B | 0.974 | 0.000 | +0.974 |
| 32 | L\_B | 0.944 | 0.705 | +0.240 |
| 32 | L\_C | 0.704 | 0.651 | +0.031 |
| 96 | L\_B | 0.910 | 0.716 | +0.194 |
| 96 | L\_C | 0.689 | 0.651 | +0.038 |

**Shared beats dedicated at every learnable capacity.** The $d' = 8$ dedicated models cannot learn the minority class at all (hard capacity floor). Representational probes (layer 8/11) reach $\sim$1.0 accuracy for both shared and dedicated.

- **Verdict**: **Negative result (no $\varepsilon$ detected).** This dual-role mechanism is a "should always stay shared" null control — analogous to $\tau \approx 0$ in the theory. The mechanism doesn't impose genuine language-divergent computation, so no $\varepsilon$ arises.

### 2.7.2 Serialisation Sharing (`Experiment7_SerializationSharing*.ipynb`)

Interchange intervention shows near-floor sharing at levels 4–5 where word order diverges, near-ceiling where it doesn't. **Evidence level:** evaluated on 3 trained checkpoints (`bp_h4_d96_s0`, `s7`, `s13`), separately, with no pooled statistics; thresholds were never frozen and there is no verdict. v2's instrument was revised after seeing results, its code-swap measurement (E1b) returns no usable numbers, and its E2 control fails. The claim rests on v1 (`Experiment7_SerializationSharing_E1_E2_E4.ipynb`, `RUN_E4_LAUNCH = False`), not on v2 (`Experiment7_v2_SerializationSharing.ipynb`). E4 calibration (A′ ceiling) was not run in v1.

- **Verdict**: Supports separation at the computation level specifically where linearisation diverges. **Suggestive only: no frozen thresholds or verdict, and no pooled statistics across the 3 checkpoints; needs replication with a noise floor before this can be considered robust.**

---

## 2.8 H-Adapter Cost Checks (Exp 8)

`Exp8_three_checks_v7.ipynb`, `Exp8_v8_valid_capacity.ipynb` (+ 6 earlier versions)

**Setup**: 10 rendering transformations tested: relabel\_new, relabel\_perm, marker, swap5, rev6, rev5, rev4, rev45, rev\_all, perm\_sym.

$a(g)$ at d = 16 ranges from 0.042 (rev6) to 0.437 (rev45).

**Check 1** (does sharing help?): **UNTESTABLE** at d = 16 — gate failed (mean gain\_A' too small). At d = 8: rev\_all does NOT share better than its parts (opposite of theory).

**Check 2** (does $a(g)$ predict sharing benefit?): Spearman($a_{d=8}$, gain\_$G_{d=8}$) = −0.13 — **no correlation**.

**d = 8 floor-corrected gain\_$G$ values** (v7):

| Rendering | gain\_$G$ | $z$-score |
|-----------|----------|----------|
| rev4 | +0.078 | +49.9 |
| perm\_sym | +0.012 | +5.9 |
| marker | +0.003 | +3.4 |
| relabel\_perm | −0.001 | −0.3 |
| swap5 | −0.003 | −2.2 |
| rev\_all | −0.014 | −7.8 |
| rev6 | −0.023 | −12.7 |
| rev5 | −0.026 | −15.6 |
| rev45 | −0.030 | −16.9 |
| relabel\_new | −0.036 | −19.4 |

rev4 shows strong sharing; most renderings show net cost. The composition rev\_all does NOT share better than its parts (rev4, rev5, rev45) — opposite of theory.

$a(g)$ stability: Spearman($a_{d=16}$, $a_{d=32}$) = 0.77 (conversion cost is a stable property of the transformation).

**Exp 6 conversion cost follow-ups** (repeated here for completeness): $a(L_C) < a(L_B)$ but $\varepsilon_C > \varepsilon_B$. Transfer test also fails.

- **Verdict**: $a(g)$ is a real, stable, measurable quantity — but it does **not predict sharing**. The conversion-cost framework does not connect to the allocation outcome. This is a theory falsification of the specific claim that $a(g)$ governs $N^*$.

**Factorial Exploratory** (`Experiment8_Factorial_Exploratory.ipynb`): CKA noise floor is high (same-model L\_A vs L\_A = 1.000 but diff-seed = 0.723). Position-matched causal interchange: anchor→L\_B retention = 0.969, anchor→L\_C retention = 0.707. Subspace principal angles at $\rho = 0.5$, $d = 96$: mean angle = 35.89° (large separation, but noise floor non-trivial).

**Numbering note**: "Exp8" labels both the H-adapter notebooks AND the factorial-exploratory notebook. The `Exp6_conversion_cost_vs_eps.ipynb` and `Exp6_transfer_test.ipynb` files are really Exp 8 follow-ups despite their filename prefix.

---

## 2.9 Hardened Substrate / Factorial (Exp 9)

`Experiment9_Factorial_Smoke.ipynb`, `Experiment9_Sanity_Degree16.ipynb`, `Experiment9b_OptionB_Smoke.ipynb`

Sanity checks pass (degree = 4, capacity-constrained regime).

**Option B** (`Experiment9b_OptionB_Smoke.ipynb`): Same grammar but L\_B has same word order as L\_A — differs only in context-keyed terminal lookup. **Single-seed** (`seeds=[0]`, 9 models total). Shared-B minus dedicated-B CE gap:

| depth | $d$ | conflicted | clean | all |
|-------|-----|-----------|-------|-----|
| 1 | 8 | +0.075 | +0.041 | +0.058 |
| 1 | 32 | +0.044 | +0.018 | +0.031 |
| 1 | 96 | +0.087 | +0.030 | +0.058 |
| 2 | 8 | +0.151 | +0.129 | +0.140 |

Shared model is WORSE than dedicated at conflicted tokens — there IS a cost to sharing when there's real terminal conflict. Gap present at all capacities and **larger at depth 2**. First synthetic setup with genuine cost-of-sharing signal driven by terminal lookup conflict ($\tau$-like), not word order / $\varepsilon$.

- **Verdict**: Single-seed smoke-test only. Option B is promising (genuine $\tau$-like signal) but not developed beyond single-seed smoke tests. **Needs multi-seed replication.**

---

# PART 3: Synthesis

---

## 3.1 Master Table of All Experiments

| Exp | Notebook(s) | Domain | Question | Result | Verdict |
|-----|-------------|--------|----------|--------|---------|
| 1 | `Experiment_1_Capacity_Competition.ipynb` | Synthetic (theory) | $N^*$ formula self-consistency | Comparative statics confirmed; symmetric case is binary | Theory consistent |
| 2 | `Experiment_2_Full.ipynb` | Synthetic (neural) | SGD peel-off order matches theory | Order matches; $2\times$ timing gap | **Partial confirmation** |
| 2-cal | `..._Revised_Calibrated_..._fixed.ipynb` | Synthetic (neural) | Calibrated $\varepsilon/\tau$ allocation | Calibration done; allocation sweep **never run** | *Incomplete* |
| 2b | `Experiment_2b_Anchor_Bias.ipynb` | Synthetic (neural) | Anchor bias produces $\varepsilon$ | Gap exists, slowly shrinks with width | *Suggestive, not conclusive* |
| 3 | `Experiment_3_Number_Agreement_PCFG.ipynb` + 5 | Synthetic (PCFG) | $\tau$ and $\varepsilon$ in PCFGs | $\tau$ works; $\varepsilon$ partially engages but no differential $N^*$ | **$\varepsilon$ mechanism engaged; $N^*$ prediction not met** |
| 4 | `Experiment4_ConflictingGender_Followups.ipynb` | Synthetic (neural) | Overlap-graded $N^*$ | Binary switch, not gradient | *Design flaw* |
| 5 | `exp5_allenzhu_paired_cfgs.ipynb` | Synthetic (neural) | Allen-Zhu $\varepsilon$ test | Never completed (compute) | *Incomplete* |
| 6 | `Experiment6_SharedTree_...CLEANED (1).ipynb` | Synthetic (neural) | Linearisation ordering of $\varepsilon$ | Ordered at small $d$; **conflicting at $d = 96$** (h4: ≈ 0; h12: positive, 2 seeds) | **Key result; A2 status configuration-dependent** |
| 6-cost | `Exp6_conversion_cost_vs_eps.ipynb` | Synthetic (neural) | $a(g)$ tracks $\varepsilon$? | Opposite ordering (2 seeds visible) | **Fail** |
| 6-xfer | `Exp6_transfer_test.ipynb` | Synthetic (neural) | Transfer follows $a(g)$? | Reverse ordering in all seeds | **Fail** |
| 6b | `Experiment6b_Syncretism_...CLEANED.ipynb` | Synthetic (neural) | Syncretism as $\varepsilon$ | Local context resolves syncretism | *Abandoned (wrong mechanism)* |
| 6b-v2 | `Experiment6b_v2_S5Relay_...ipynb` | Synthetic (neural) | S5-relay substrate | Architecture cliff | *Abandoned* |
| 7 | `Experiment7_DualRole_tAgree_CLEANED.ipynb` | Synthetic (neural) | Dual-role $\varepsilon$ | Shared beats dedicated at all capacities | **Null control** ($\tau \approx 0$) |
| 7-ser | `Experiment7_SerializationSharing*.ipynb` | Synthetic (neural) | Interchange at linearisation divergence | Separation where order diverges (3 checkpoints evaluated separately; no frozen thresholds, no verdict; v1) | Suggestive; needs replication |
| 8 | `Exp8_three_checks_v7.ipynb`, `Exp8_v8_...ipynb` | Synthetic (neural) | $a(g)$ predicts sharing? | Spearman = −0.13 | **Theory falsified** ($a(g) \not\Rightarrow N^*$) |
| 9 | `Experiment9*.ipynb` (3 files) | Synthetic (neural) | Hardened substrate | Single-seed smoke-test: $\tau$-like signal in Option B | *Smoke only (single-seed)* |
| 10 | `Exp10_Gemma_IOI_Representation.ipynb` | Natural (Gemma) | Variance-PCA subspace | No effect | *Instrument inadequate* |
| 11 | `Exp11_Gemma_IOI_CausalSubspace.ipynb` | Natural (Gemma) | Gradient subspace, mean-ablation | Positive control fails | *Instrument inadequate* |
| 12 | `Exp12_Gemma_IOI_Erasure.ipynb` | Natural (Gemma) | Projection erasure | Partial fix (FR/YO pass, EN fails) | *Partial instrument fix* |
| 13 | `Exp13_Gemma_IOI_Scale.ipynb` | Natural (Gemma) | IOI head sharing at scale | Sharing confirmed, decreases with scale | **Valid positive** |
| 14 | `Exp14_HeadOutput_Patching.ipynb` | Natural (Gemma) | Head-output language-agnosticism | Compat $\approx 1.0$ all scales | **Cleanest positive result** |
| 15 | `Exp15_Agreement_Scale.ipynb` | Natural (Gemma) | Agreement baselines | Baselines only | *Incomplete* |
| 15b | `Exp15b_Agreement_MultiLang.ipynb` | Natural (Gemma) | Agreement sharing at scale | Sharing increases with scale (compat > 1, partly denominator artifact) | **Contradicts $\varepsilon/N^*$** |
| 15-geom | `Exp15_IOI_HeadWrite_Geometry.ipynb` | Natural (Gemma) | Head write geometry | Not executed | *Abandoned* |
| 16 | `Exp16_v3_Residual_Stream_Patching.ipynb` | Natural (Gemma) | Depth of sharing split | No clean split (FR/DE); HI confounded | *Unresolved* |
| 17 | `Exp17_Rep_Space_Divergence.ipynb` | Natural (Gemma) | Representation divergence | HI diverges; FR/DE don't | Valid — $\varepsilon > 0$ distant |
| 18 | `Exp18_Preflight_BLOOM.ipynb` | Natural (BLOOM) | IOI gate | All languages pass | Gate passed |
| 19 | `Exp19_Preflight_Tasks.ipynb` | Natural (BLOOM) | Task gates | Mixed | Gate results |
| 20 | `Exp20_Preflight_Round2.ipynb` | Natural (BLOOM) | Round 2 gates | Token collisions in word order | *Design flaw* (tokenizer) |
| 21 | `Exp21_Computation_vs_Representation.ipynb` | Natural (BLOOM) | Computation sharing | IOI shared ($C_\text{resid} \approx 1.0$) | **Valid — IOI sharing replicates** |
| 22 | `Exp22_Translation_WordOrder_Patching.ipynb` | Natural (BLOOM) | Translation vs. word order | Translation shared; word order NOT | **First clean separation** |
| 23 | `Exp23_Representation_Wu_Lim.ipynb` | Natural (BLOOM) | Representation alignment | Aligns with patching | Convergent evidence |
| 24 | `Exp24_Concept_Gender_Feature_vs_Lookup.ipynb` | Natural (Gemma) | Gender concept gate | All pass | Gate passed |
| 25 | `Exp25_WordOrder_Parallel.ipynb` | Natural (BLOOM/Gemma) | Word order CKA | Language-specific | Confirms Exp 22 |
| 26 | `Exp26_WordOrder_Calibrated.ipynb` | Natural (Gemma) | Word order DTI v1 | CIs include 0 | *Superseded by Exp 27* |
| 27 | `Exp27_A_WordOrder_Gate.ipynb` | Natural (Gemma) | Word order ceiling DTI | 10/15 WALL, 1 TRANSFER | **INVERSION\_ABANDONED** |
| 28 | `Exp28_B_Gender_DTI.ipynb` | Natural (Gemma) | Gender DTI | Only 2 viable languages | **INSTRUMENT\_INADEQUATE** |
| 29 | `Exp29_DirectionTransfer_DAS_v5.ipynb` | Natural (Gemma) | DAS direction transfer (9B-it only) | Gender: 11/12 TRANSFER; Order: 6/7 WALL | **Strongest asymmetry evidence** |

---

## 3.2 Recurring Failure Modes

### Mode A: $\varepsilon$ Cannot Be Reliably Operationalised

The single most consequential failure across the project. Every attempt to inject or measure $\varepsilon$ in a controlled synthetic setting encountered a different obstacle:

| Exp | $\varepsilon$ operationalisation | Outcome |
|-----|----------------------------------|---------|
| 3 | Lookup-table divergence (PCFG) | Mechanism partially engages (eps\_cost > 0 at cap 6–8) but model bypasses before differential $N^*$ emerges |
| 4 | Vocabulary overlap | Binary switch, no gradient |
| 6 (h4) | Linearisation distance | Ordered at small $d$; fades to $\sim 0$ by $d = 96$ |
| 6 (h12) | Linearisation distance | Ordered at small $d$; **persists at $d = 96$** (2 seeds) |
| 6-cost | Conversion cost $a(g)$ | Ordering opposite to $\varepsilon$ |
| 6b | Terminal syncretism | Tests input destruction, not anchor bias |
| 7 | Dual-role frequency | Zero signal; null control |
| 2b | Training imbalance | Suggestive but gap keeps shrinking |
| 8 | H-adapter cost | $a(g)$ does not predict sharing |

**Assessment**: The status of $\varepsilon$ is genuinely configuration-dependent. Under h12, the linearisation design (Exp 6) produces persistent, reproducible $\varepsilon$ at d=96 across 2 seeds. Under h4, it does not. No operationalisation achieves the full predicted chain: $\varepsilon$ → differential $N^*$ → allocation shift. But the partial engagements (Exp 3 cap=6-8, Exp 6 h12) suggest the concept has SOME empirical grip, even if the framework's quantitative predictions fail.

### Mode B: Instrument / Gate Failures

Many planned comparisons were eliminated by the measurement instrument failing its own positive-control gate:

| Exp | Instrument | Gate failure |
|-----|-----------|-------------|
| 10–12 | Subspace erasure (PCA, gradient, projection) | EN positive control fails in all variants |
| 15b | Agreement head patching | AR (tokenizer collision — all verb pairs dropped), SW (low accuracy), HI (accuracy at scale) |
| 20 | Word order | first\_token\_collision = 1.0 |
| 27 | DTI word order | Ceiling control inverts in 10/15 cells |
| 28 | DTI gender | Only 2 viable recipient languages (AR passes accuracy + ceiling but fails DTI coherence) |

**Assessment**: These are not theory falsifications — the theory was never tested where the instrument failed. But the pattern is informative: causal-intervention instruments on real LLMs have a high failure rate, especially for (a) tokenizer-sensitive tasks on non-European languages and (b) tasks requiring fine-grained behavioural flips (word order).

### Mode C: Experiments Never Completed

| Exp | What was missing | Likely reason |
|-----|-----------------|---------------|
| 2-cal | Full allocation sweep | Not documented |
| 4 | Disjoint-vocabulary control | Sweep interrupted |
| 5 | Full 60-run sweep + $\varepsilon$ analysis | Compute ($\sim$100–150 A100-hours) |
| 15 | Head ranking and patching | Not documented (cells commented out) |
| 15-geom | Main computation cells | Not documented |

---

## 3.3 What Survived

Despite the framework's overall negative trajectory, several empirical findings are robust and independently valuable:

### Finding 1: IOI Circuits Are Shared Across Languages and Scales

**Evidence**: Head ablation (Exp 13), head-output patching (Exp 14), residual-stream patching (Exp 21). Replicated across Gemma-2 2B/9B/27B and BLOOM 1b7/7b1.

**Strength**: Exp 14 compatibility $\approx 1.0$ is the cleanest result — EN head outputs are perfectly substitutable for FR/YO outputs.

### Finding 2: Word Order Is Language-Specific

**Evidence**: frac\_flip $\approx 0$ in BLOOM patching (Exp 22), low CKA for distant pairs (Exp 25), 6/7 WALL cells in direction transfer (Exp 29 v5).

**Strength**: Converging evidence across models, methods, and language pairs. The one exception (Exp 27 finding 1/15 TRANSFER) is anomalous rather than refuting.

### Finding 3: Gender Directions Transfer Cross-Linguistically

**Evidence**: 11/12 gender cells = TRANSFER in Exp 29 v5. FR→ES ($R = 1.06$), FR→HI ($R = 0.56$), FR→AR ($R = 0.94$). Direction is additive, necessary, specific, and transfers across language families.

### Finding 4: The Task-Dependent Sharing Asymmetry

Gender: shared. Word order: separated. IOI: shared. Translation: shared. Agreement: shared (and increasingly so with scale).

This is the project's most reliable high-level finding. It does not require the $N^*/\varepsilon$ framework to state — it is a descriptive empirical regularity about which linguistic functions share circuits in multilingual LLMs.

### Finding 5: Representation and Computation Are Dissociable

Exp 17 shows HI representations diverge from EN ($+0.290$ corrected cosine divergence) while Exp 14 shows computation is shared (compatibility $\approx 1.0$). A model can share circuits while encoding language-specific representational geometry.

### Finding 6: $\tau$ Works in Synthetic Settings

Across Exp 1 (theory), Exp 2 (neural), and Exp 3 (PCFG), the $\tau$ parameter (shortcut richness) reliably produces capacity-dependent separation. Languages with high $\tau$ specialise earlier. This half of the framework is consistent with all evidence.

---

## 3.4 What Is Falsified vs. Unresolved

### Falsified

1. **$a(g)$ governs $N^*$**: Exp 8 shows Spearman = −0.13 between conversion cost and sharing gain. Exp 6-cost shows $a(L_C) < a(L_B)$ with $\varepsilon_C > \varepsilon_B$. The specific claim that conversion cost predicts allocation is falsified.

2. **$\varepsilon$-as-lookup-divergence produces differential $N^*$**: Exp 3 tested this across 4 PCFG designs. $\varepsilon$ partially engages (positive eps\_cost at cap 6–8) but never produces differential $N^*$. The mechanism is understood (model bypasses the manipulation before the threshold shifts).

3. **Sharing monotonically decreases with scale** (the $N^*$ survival-curve prediction): Exp 15b shows agreement sharing **increases** with scale for close languages. This directly contradicts the framework's central prediction. (Qualification: compat >> 1 at 27B is partly a denominator artifact, but the DIRECTION — increasing, not decreasing — is clear.)

### Specific mechanisms failed; framework-level test incomplete

The $N^*/\varepsilon$ framework as formulated has lost its two main operational predictions: $a(g) \rightarrow N^*$ is falsified, and monotonic scale decrease is contradicted by agreement. But the framework's core assumption — that $\varepsilon$ exists as an irreducible floor (A2) — is **genuinely unresolved**, not confirmed or refuted:

- Under h4 (Exp 6 ConvergenceCheck), $\varepsilon$ fades to ~0 at d=96 → A2 violated.
- Under h12 (Exp 6 main notebook, step-matched, 2 seeds), $\varepsilon$ persists at d=96 → A2 not clearly violated.
- The best-designed test (Exp 5, Allen-Zhu CFGs) was never completed.
- Exp 2b (anchor bias) is suggestive but the gap keeps shrinking.

The correct characterisation is: specific mechanisms predicted by the framework are falsified; whether the core conceptual claim ($\varepsilon$ as an irreducible capacity-independent floor) holds in neural networks remains untested at the required standard.

### Unresolved

1. **Does $\varepsilon$ exist as an irreducible floor?** Configuration-dependent in Exp 6 (h4 says no, h12 says maybe). Exp 5 (Allen-Zhu) was never run. Exp 2b is suggestive but not conclusive.

2. **Why does sharing increase with scale for agreement but decrease for IOI?** Exp 13 shows head-ablation sharing weakening from 2B to 27B (retained rises). Exp 15b shows agreement sharing strengthening. No explanation is documented.

3. **What drives word-order separation?** Exp 22, 25, 27, 29 consistently show word order is language-specific, but the causal mechanism is not identified. The linearisation result (Exp 6, 7-ser) suggests it relates to how attention mechanisms handle different child orderings, but this is a reframing, not an explanation.

4. **Depth of the sharing/separation transition**: Exp 16 found no clean split depth for close languages (FR, DE). Whether the transition happens at a single critical layer or is distributed remains unresolved.

5. **The h4/h12 discrepancy at d=96**: Why does n\_head=4 show $\varepsilon \approx 0$ while n\_head=12 shows persistent positive $\varepsilon$ on the same PCFG, same capacity, same training? This is the single most important unresolved question in the synthetic experiments.

---

## 3.5 Contradictions and Anomalies

1. **Exp 6 h4 vs. h12 at d=96**: The same design with different head counts gives opposite conclusions about whether $\varepsilon$ persists. h4 (ConvergenceCheck, 2 seeds): CE-gap B ≈ −0.009. h12 (main notebook, step-matched, 2 seeds): eps\_B = +0.023–0.046, eps\_C = +0.060–0.064. Both are in the repository; neither is obviously wrong. This is NOT a convergence-matching artifact — the h12 results include explicit step-matching and lockstep extension.

2. **Exp 6 vs. Exp 6-cost**: At small $d$, $\varepsilon_C > \varepsilon_B$ from the main experiment but $a(L_C) < a(L_B)$ from the conversion-cost measurement ($a(L_B) \approx 0.045$, $a(L_C) \approx 0.000$). Full reversal is trivially cheap to *convert* (just reverse) but expensive to *share*. The conversion-cost metric measures the wrong thing.

3. **Exp 13 vs. Exp 15b**: IOI sharing weakens with scale; agreement sharing strengthens with scale. Both use head-level interventions on the same model family. The difference must be task-dependent but the mechanism is not documented.

4. **Exp 27: 1 TRANSFER in 15 cells**: If word order is truly language-specific, why does one cell show transfer? Could be noise, could be a genuine exception. The notebook labels the entire experiment INVERSION\_ABANDONED rather than resolving this anomaly.

5. **Exp 29 v5 vs. Exp 28**: Exp 29 finds gender transfer robustly; Exp 28 (same model, same feature type) finds INSTRUMENT\_INADEQUATE. The instruments differ (DAS-lite vs. DTI) but the discrepancy raises questions about whether the instrument determines the finding.

---

## 3.6 Superseded / Duplicate Notebooks

The repository contains numerous superseded versions. For the record:

| Current version | Superseded by | Why |
|----------------|--------------|-----|
| `Experiment_1_Capacity_Competition (1).ipynb` | `Experiment_1_Capacity_Competition.ipynb` | Identical copy |
| `Experiment_2_Learnable_Allocation.ipynb` (×2) | `Experiment_2_Full.ipynb` | Early incomplete version |
| `Experiment_2_Learnable_Shared_Private_Allocation.ipynb` | `Experiment_2_Full.ipynb` | Less thorough variant |
| `Experiment_2_Revised_Calibrated_Shared_Private.ipynb` | `..._fixed.ipynb` | Code bug (tuple unpacking) |
| `Experiment4_ConflictingGender_Followups (1).ipynb` | Main copy | Identical, no outputs |
| `Experiment7_DualRole_tAgree.ipynb` (4 variants) | `..._CLEANED.ipynb` | Progressive iterations |
| `Experiment8_adapter_cost.ipynb` | `Exp8_three_checks_v7.ipynb` | Earlier d=32 version, partially complete (6/11 translators, 0/20 joint) |
| `Exp8_three_checks.ipynb` v1–v6 | v7 | Progressive iterations |
| `Exp16_Residual_Stream_Patching.ipynb` v1, v2 | v3 | Instrument refinement |
| `Exp26_WordOrder_Calibrated.ipynb` | Exp 27 | Three instrument problems found |
| `Exp29_DirectionTransfer_DAS.ipynb` v1–v4 | v5 | Progressive instrument refinement |

**Numbering collision**: "Exp8" labels both the H-adapter cost-check notebooks (`Exp8_three_checks_*.ipynb`) and the factorial-exploratory notebook (`Experiment8_Factorial_Exploratory.ipynb`). Files `Exp6_conversion_cost_vs_eps.ipynb` and `Exp6_transfer_test.ipynb` are thematically Exp 8 follow-ups despite the Exp 6 prefix.

---

## 3.7 Uncertainties

1. **SMOKE mode artifacts**: Exp 10–12 ran in SMOKE mode (5 prompts/language). Numbers may shift with full prompt sets; the qualitative instrument-failure pattern is unlikely to change.

2. **The h4/h12 discrepancy at d=96**: The most consequential unresolved uncertainty in the synthetic experiments. The ConvergenceCheck (h4, n\_head=4) shows $\varepsilon \approx 0$ at d=96 through 180K steps. The main notebook's step-matched diagnostic (h12, n\_head=12) shows clearly positive $\varepsilon$ at d=96, reproducible across 2 seeds with explicit step-matching (cell 47 equalises step counts, then extends both by +30K in lockstep). These are different architectures on the same grammar. The difference could reflect how attention routing changes with head count, but no notebook investigates this. Every downstream conclusion about A2 depends on which configuration is considered canonical.

3. **Seed count**: Exp 6 main uses 2 seeds at d=96 (h12); Exp 6 ConvergenceCheck uses 2 seeds (h4); Exp 8 uses 3 at d=8 only. For the critical $\varepsilon$-ordering results, 2 seeds establishes sign agreement but not quantitative robustness. Exp 7-ser was evaluated on 3 trained checkpoints (bp_h4_d96_s0, s7, s13) separately, with no pooled statistics, frozen thresholds or verdict (v2's instrument was revised after seeing results; its E1b returns no usable numbers and its E2 control fails; the claim rests on v1). Exp 9b is single-seed (`seeds=[0]`).

4. **IT vs. base models**: Exp 29 v5 uses Gemma-2-9B-**it** (instruction-tuned). Whether direction-transfer results hold on base models is untested. Instruction tuning could strengthen or weaken cross-language feature alignment.

5. **Tokenizer as confound**: AR, SW, HI, and ZH comparisons are compromised by tokenizer issues (collisions, high fertility, script differences) in at least 4 experiments. The project's strongest results (Exp 14, 22, 29) are on European languages plus YO — the typological range is narrower than it appears.

6. **The framework assessment**: The working assessment that the framework's specific mechanisms ($a(g) \rightarrow N^*$, monotonic scale decrease) have failed is well-supported. The stronger claim that $\varepsilon$ itself is "not rescuable" depends on the h4 reading of Exp 6; the h12 reading leaves A2 open. The Allen-Zhu experiment (Exp 5) — the best-designed test of whether $\varepsilon$ exists as an irreducible floor — was never completed. A future completion could change the picture.

7. **Security**: `Exp16_v3_Residual_Stream_Patching.ipynb` contains an exposed HF\_TOKEN in a code cell. Must be scrubbed before any publication.

8. **Exp 15b denominator issue**: Compat values >> 1 (especially 4.32 at 27B FR) reflect small denominators (EN-ranked heads account for only 23% of FR agreement logit-diff at that scale), not genuinely enhanced sharing. The directional claim (sharing increases, contradicting theory) is robust, but the magnitudes should not be interpreted literally.

---

## Notebook-to-Section Index

| Notebook | Section |
|----------|---------|
| `Experiment_1_Capacity_Competition.ipynb` | 2.1 |
| `Experiment_2_Full.ipynb` | 2.2.1 |
| `Experiment_2_Revised_Calibrated_Shared_Private_fixed.ipynb` | 2.2.2 |
| `Experiment_2b_Anchor_Bias.ipynb` | 2.2.3 |
| `Experiment_3_Number_Agreement_PCFG.ipynb` | 2.3 |
| `PCFG_git.ipynb`, `v8_multilingual_pcfg.ipynb`, `v9_multilingual_pcfg.ipynb` | 2.3 |
| `Experiment4_ConflictingGender_Followups.ipynb` | 2.4 |
| `exp5_allenzhu_paired_cfgs.ipynb` | 2.5 |
| `Experiment6_SharedTree_DivergentLinearization_CLEANED (1).ipynb` | 2.6.1 |
| `Experiment6_ConvergenceCheck_d96.ipynb` | 2.6.1 |
| `Exp6_conversion_cost_vs_eps.ipynb` | 2.6.2 |
| `Exp6_transfer_test.ipynb` | 2.6.3 |
| `Experiment6b_Syncretism_BudgetMatched_CLEANED.ipynb` | 2.6.4 |
| `Experiment6b_v2_S5Relay_ValidationLadder.ipynb` | 2.6.5 |
| `Experiment7_DualRole_tAgree_CLEANED.ipynb` | 2.7.1 |
| `Experiment7_SerializationSharing*.ipynb` | 2.7.2 |
| `Exp8_three_checks_v7.ipynb`, `Exp8_v8_valid_capacity.ipynb` | 2.8 |
| `Exp8_relabel_circuit_construction.ipynb` | 2.8 (side-branch) |
| `Experiment8_adapter_cost.ipynb` | 2.8 (superseded) |
| `Experiment8_Factorial_Exploratory.ipynb` | 2.8 (factorial) |
| `Experiment9*.ipynb` | 2.9 |
| `Exp10_Gemma_IOI_Representation.ipynb` | 1.1.1 |
| `Exp11_Gemma_IOI_CausalSubspace.ipynb` | 1.1.1 |
| `Exp12_Gemma_IOI_Erasure.ipynb` | 1.1.1 |
| `Exp13_Gemma_IOI_Scale.ipynb` | 1.1.1 |
| `Exp14_HeadOutput_Patching.ipynb` | 1.1.2 |
| `Exp15_Agreement_Scale.ipynb` | 1.2.1 |
| `Exp15b_Agreement_MultiLang.ipynb` | 1.2.2 |
| `Exp15_IOI_HeadWrite_Geometry.ipynb` | 1.2 (abandoned) |
| `Exp16_v3_Residual_Stream_Patching.ipynb` | 1.3 |
| `Exp17_Rep_Space_Divergence.ipynb` | 1.1.3 |
| `Exp18_Preflight_BLOOM.ipynb` | 1.4.1 |
| `Exp19_Preflight_Tasks.ipynb` | 1.4.1 |
| `Exp20_Preflight_Round2.ipynb` | 1.4.1 |
| `Exp21_Computation_vs_Representation.ipynb` | 1.4.2 |
| `Exp22_Translation_WordOrder_Patching.ipynb` | 1.4.3 |
| `Exp23_Representation_Wu_Lim.ipynb` | 1.4.4 |
| `Exp24_Concept_Gender_Feature_vs_Lookup.ipynb` | 1.5.1 |
| `Exp25_WordOrder_Parallel.ipynb` | 1.5.2 |
| `Exp26_WordOrder_Calibrated.ipynb` | 1.5.3 |
| `Exp27_A_WordOrder_Gate.ipynb` | 1.5.3 |
| `Exp28_B_Gender_DTI.ipynb` | 1.5.4 |
| `Exp29_DirectionTransfer_DAS_v5.ipynb` | 1.5.5 |
