# Shared-Grammar PCFG Experiments — Summary

Three experiments testing when a multilingual LM builds a **shared** circuit
vs. a **language-specific** one, using a synthetic PCFG with three "languages"
(A/B/C) that are different linearizations of the *same* underlying tree. The
theory predicts a crossover capacity `N*`, set by two costs: `τ` (extra
compute a shared circuit needs when languages' computations differ) and `ε`
(the irreducible cost of routing a non-anchor language through an
anchor-shaped shared representation).

---

## Experiment 6 — Shared-tree, divergent-linearization ε

**Tested:** Does cross-lingual sharing cost more at low capacity when
languages differ only in *word order*? L_A reads the tree in canonical
order, L_B reverses two of six grammar levels, L_C reverses all six —
same tokens, same tree, same vocabulary across all three; only the child
order read-out differs. Trained a shared model on a 70/15/15 (A/B/C) mix
and compared it to disjoint (per-language) models via `permute_grammar`
(matched rule-length/branching stats, not just a fresh random grammar).

**Result:** Big-pool sweep (500K trees, d=8–96): co-training cost ordered
exactly as predicted by word-order distance — level 4 > level 5 > level 6
(≈0) — and it **fades to ≈0 by d=96**. No persistent ε floor. All
representation-geometry instruments tried (subspace overlap, probe
transfer, ablation) were noisy/confounded; the functional measures (CE
gap, readout ceiling) were the clean signal.

**Why it fades:** diagnosed as a design gap, not a measurement failure —
every language shares identical tokens/embeddings, so there's no
upstream bias giving ε anywhere to live. → motivated **Experiment 6b**
(add syncretism to create a real information-loss asymmetry).

---

## Experiment 6b — Syncretism + budget-matched dedicated models

**Tested:** Two fixes to Exp 6. (1) **Syncretism** — collapse two
terminals onto the same surface token for L_B/L_C only (`eps_level=1`
mild, `=2` strong), creating genuine embedding-level information loss
that capacity can only partially undo. (2) **Budget-matched** dedicated
baselines — train K=3 dedicated models at width d′ so `3×params(d′) ≈
params(d)`, instead of comparing against full-width solo models (~3x more
params than fair).

**Result (eps_level=1):** terminal-recovery probe and CE-gap both showed a
small "sharing wins" signal, but it didn't hold up:
- d=96 runs were **still improving at 90K steps** — not converged, so
  every d=96 number here is provisional.
- A parameter-ratio check showed the budget-match mismatch alone could
  explain ~63% of the apparent gap at d=32; only d=16 is mostly clean.
- A follow-up pre-flight audit found **31–39% of collapsed positions are
  genuinely unresolvable** even at a 32-token context window (the
  original <1%-ambiguous number was misleading — it averaged in trivially
  unique, never-repeated contexts).

**Why it was abandoned:** re-reading the cost derivation showed
terminal-collapse syncretism isn't actually a test of the paper's ε — it
destroys information at the *input*, which a dedicated (non-shared)
circuit would suffer identically. Any gap found is evidence for
source-side information loss, not anchor-bias — a related but different
cost. → tried an S5-relay (group-composition) redesign to get a
genuinely non-shortcuttable mechanism, abandoned before training in favor
of a simpler, **lossless** idea → **Experiment 7**.

---

## Experiment 7 — t2 dual-role ε test

**Tested:** A lossless redesign. One terminal (`t2`) plays one of two
grammatical roles (ROLE-X / ROLE-Y) depending on which rule produced it —
always 100% recoverable from tree structure, never destroyed. L_A's own
rule statistics make `t2` mostly ROLE-X; L_B/L_C's permuted rule sets make
it mostly ROLE-Y (~90/10 skew either way). The measured task,
`t_agree`, is a downstream agreement token that depends on `t2`'s role,
forcing the model to actually propagate role information from `t2`'s
embedding rather than reading it off local context.

**Result (after a long chain of eval bugs — divide-by-zero masquerading
as a 0.000 floor, a per-language reweighting bug, a role-probe built on
the wrong label distribution, each initially looking like a real
finding):** at the two capacities where the task is actually learnable
(`d=32/d'=16` and `d=96/d'=56` — `d'=8` is a hard capacity floor nobody
clears), **the shared model beats or matches the dedicated one** for both
languages: L_B +0.240 (d=32) / +0.194 (d=96), L_C +0.031 / +0.038.

**Conclusion:** no anchor-bias ε detected for this mechanism at any
learnable capacity — a genuine **negative/null result**, useful as the
"should always stay shared" control at the opposite end of the (τ, ε)
spectrum from Experiment 6's positive word-order finding. A follow-up
(Exp 7-B) tried to make the propagation itself language-divergent
(distance injection, relay hops, boolean composition) so the null result
couldn't be guaranteed by construction, but every variant was judged
solvable by a transformer "almost for free" and abandoned before
training. The project then pivoted to a **structural-axes** design
(ordering / feature-agreement / dependency-nesting) on a hardened shared
grammar, which is where the experimental line currently stands.
