# Shared vs. separate circuits in multilingual LLMs

## Start here

If you read nine notebooks, read these, in this order:

1. [`Exp14_HeadOutput_Patching`](natural_data/gemma_ioi_agreement/Exp14_HeadOutput_Patching.ipynb): the cleanest positive result. English IOI head outputs patched into French and Yoruba runs recover about 100% of the effect at every Gemma-2 scale, so the shared heads write in a language-agnostic format.
2. [`Exp21_Computation_vs_Representation`](natural_data/bloom_task_grid/Exp21_Computation_vs_Representation.ipynb): the same shared-IOI result on BLOOM, even for a very distant pair (EN→AR residual-patching compatibility ≈ 0.98), and the harness that separates shared computation from shared representation.
3. [`Exp22_Translation_WordOrder_Patching`](natural_data/bloom_task_grid/Exp22_Translation_WordOrder_Patching.ipynb): the first clean separation. Translation is shared (compatibility 0.84–1.13); word order is not (flip fraction 0.03–0.19).
4. [`Exp29_DirectionTransfer_DAS`](natural_data/direction_transfer/Exp29_DirectionTransfer_DAS.ipynb): a gender direction transfers across languages (11 of 12 cells) while a word-order direction does not (6 of 7 walled), and each is specific to its own feature. One exploratory run on Gemma-2-9B-it.
5. [`Exp06_SharedTree_DivergentLinearization`](synthetic/shared_tree_linearization/Exp06_SharedTree_DivergentLinearization.ipynb): the toy that reproduces the word-order result. One abstract tree rendered in three word orders; at small capacity the shared model pays extra loss ordered by reordering distance. At d=96 the outcome depends on head count (open).
6. [`Exp07b_Serialization_Sharing_v1`](synthetic/shared_tree_linearization/Exp07b_Serialization_Sharing_v1.ipynb): interchange tests on that toy show near-floor sharing at the tree levels where word order diverges and near-ceiling where it does not. The report's claim rests on this version (v1). Evidence level: evaluated on 3 trained checkpoints (`bp_h4_d96_s0`, `s7`, `s13`), separately, no pooled statistics; thresholds never frozen, no verdict. Treat as suggestive, not a result.
7. [`Exp08c_Conversion_Cost_vs_Eps`](synthetic/conversion_cost/Exp08c_Conversion_Cost_vs_Eps.ipynb): a failed mechanism. The cost of converting one word order into another does not track the sharing cost: the fully reversed language is cheaper to convert but costs more to share.
8. [`Exp08d_Transfer_Test`](synthetic/conversion_cost/Exp08d_Transfer_Test.ipynb): a second failed prediction. Lower conversion cost did not mean faster adaptation; the reverse ordering held in all 3 seeds. Reads results written by Exp08c.
9. [`Exp03_Number_Agreement_PCFG`](synthetic/pcfg_agreement/Exp03_Number_Agreement_PCFG.ipynb): why the toy for gender and agreement is missing. A cheaper shortcut reduces the capacity needed, but the model bypassed every conflict design, so none shifted the capacity at which the task is solved.

Research notebooks asking whether a multilingual model handles the same function with one shared circuit or one per language. Everything here is a saved run: notebooks carry their outputs and are read, not rerun.

**Summary**
1. On real models (Gemma-2 2B–27B, BLOOM 1b7/7b1) sharing is task-dependent: IOI, translation and negation are shared across languages, while word order is separated, even between languages that share the same order.
2. Gender and agreement fall in between and depend on how close the languages are; Hindi is the weakest case, and Hindi gender is the one open conflict between BLOOM and Gemma.
3. A toy model with one abstract tree rendered in different word orders reproduces the word-order result: the shared model pays where the orders diverge and shares where they do not.
4. Mechanisms that failed: subspace erasure as a sharing probe, conversion cost as a predictor of sharing, a monotone decrease of sharing with scale, and every toy conflict design that gave the model a bypass.
5. The strongest direction-transfer evidence (Exp29) is one exploratory run on one instruction-tuned model; the toy interchange results (Exp07b) have no frozen thresholds or verdict, and the Option B smoke test (Exp09b) is single-seed. See [Known limitations](#known-limitations).

Short report: [`docs/report.md`](docs/report.md). Full report with every experiment, number and caveat: [`docs/report_full.md`](docs/report_full.md).

## How to read this repo

Read Part 1 (natural data) first, then Part 2 (toy models). Within a folder, read in experiment-number order.

```
natural_data/                    PART 1: real LLMs
  gemma_ioi_agreement/           Exp13-17   IOI head ablation/patching, agreement, depth patching, representation
  bloom_task_grid/               Exp18-23   BLOOM preflight gates, task-grid patching, representation analysis
  direction_transfer/            Exp24-29   gender concept, word order, direction transfer
synthetic/                       PART 2: toy models
  allocation_toy/                Exp01-02b  closed-form and learnable allocation
  pcfg_agreement/                Exp03-03a  PCFG number agreement
  conflicting_gender/            Exp04      conflicting gender
  paired_cfgs/                   Exp05      paired grammars (designed, not run)
  shared_tree_linearization/     Exp06-07b  shared tree, different word orders; dual role; serialization sharing
  conversion_cost/               Exp08a-d   conversion-cost and adaptation tests
  hardened_substrate/            Exp09-09b  harder grammar: exploratory grid, sanity check, Option B smoke test
archive/                         superseded versions, duplicates, smoke tests, abandoned designs (see archive/README.md)
docs/                            report.md, report_full.md, RENAMES.md (old name -> new name)
```

Numbering note: the old `Exp8` name was used for two different things. The conversion-cost tests are now **Exp08a-d** (including two files previously named `Exp6_*`), and the exploratory grid is filed with Exp09.

## Kept notebooks

Results are copied from the reports; "Report §" refers to [`docs/report_full.md`](docs/report_full.md).

### Part 1: natural data

| Notebook | Model / substrate | Question | Key result | Seeds / evidence | Report § |
|---|---|---|---|---|---|
| [`Exp13_Gemma_IOI_Scale`](natural_data/gemma_ioi_agreement/Exp13_Gemma_IOI_Scale.ipynb) | Gemma-2 2B / 9B / 9B-it / 27B-it; EN, FR, YO | Do IOI heads ranked in English matter for FR and YO? | Ablating the English-ranked heads hurts FR/YO at every scale (retained logit-diff 0.17–0.35 at 2B, 0.54–0.74 at 27B-it). Sharing weakens with scale but persists. The subspace-erasure track never passed its own gate. | Full run, 4 models | 1.1.1 |
| [`Exp14_HeadOutput_Patching`](natural_data/gemma_ioi_agreement/Exp14_HeadOutput_Patching.ipynb) | Gemma-2, same 4 models | Do the shared heads write in a language-agnostic format? | Patching English head outputs into FR/YO runs recovers about 100% (cross-compatibility 0.87–1.38; same-language floor ≈ 1.0) at all scales. Cleanest positive result. | Full run, 4 models | 1.1.2 |
| [`Exp15b_Agreement_MultiLang`](natural_data/gemma_ioi_agreement/Exp15b_Agreement_MultiLang.ipynb) | Gemma-2 2B–27B; FR, DE (HI, AR, SW lost) | Does subject–verb agreement stay shared with scale? | For FR/DE cross-language compatibility rises with scale. Large values (4.32 at 27B FR) partly reflect a small denominator: direction reliable, magnitude not. AR (tokenizer collision), SW and HI failed gates. | Gate failures in 3 languages | 1.2.2 |
| [`Exp16_Residual_Stream_Patching`](natural_data/gemma_ioi_agreement/Exp16_Residual_Stream_Patching.ipynb) | Gemma-2 2B / 9B / 27B; EN→FR, DE, HI | At which depth do residual states stop being interchangeable? | No clean split depth for FR/DE (partial verdicts); HI split at layer 0 is confounded by script. Unresolved. | Instrument unresolved | 1.3 |
| [`Exp17_Rep_Space_Divergence`](natural_data/gemma_ioi_agreement/Exp17_Rep_Space_Divergence.ipynb) | Gemma-2 9B; EN, FR, DE, HI | Do representations diverge even where computation is shared? | Corrected divergence: HI +0.29, DE +0.00, FR −0.06, while Exp14 shows the computation is shared. Representation and computation come apart for the distant language. | One model | 1.1.3 |
| [`Exp18_Preflight_BLOOM`](natural_data/bloom_task_grid/Exp18_Preflight_BLOOM.ipynb) | BLOOM 1b7 / 7b1; 7 languages | Can the IOI task be measured in each language? | IOI accuracy gate passes for all 7 languages on both models. Gate notebook, no sharing result. | Gate only | 1.4.1 |
| [`Exp19_Preflight_Tasks`](natural_data/bloom_task_grid/Exp19_Preflight_Tasks.ipynb) | BLOOM 1b7 / 7b1 | Which of negation, antonym, agreement, gender, classifier are measurable per language? | Mixed pass/fail; fixes the set of task × language cells later notebooks may use. | Gate only | 1.4.1 |
| [`Exp20_Preflight_Round2`](natural_data/bloom_task_grid/Exp20_Preflight_Round2.ipynb) | BLOOM 1b7 / 7b1 | Gates for word order, translation, ZH classifier, AR numeral polarity | Word-order candidates collide on their first token for some language pairs, a tokenizer-level design limit. | Gate only | 1.4.1 |
| [`Exp21_Computation_vs_Representation`](natural_data/bloom_task_grid/Exp21_Computation_vs_Representation.ipynb) | BLOOM; 7 languages | Is the computation shared, and is the representation shared, on the same items and layers? | IOI sharing replicates on BLOOM (residual-patching compatibility ≈ 0.98 for EN→AR on bloom-1b7), including a very distant pair. | Patching + probes, 2 models | 1.4.2 |
| [`Exp22_Translation_WordOrder_Patching`](natural_data/bloom_task_grid/Exp22_Translation_WordOrder_Patching.ipynb) | BLOOM 7b1 | Does translation share circuits more than word order does? | Translation shared (compatibility 0.84–1.13); word order not shared (flip fraction 0.03–0.19). First clean separation in the natural-data work. | Single model; gated cells only | 1.4.3 |
| [`Exp23_Representation_Wu_Lim`](natural_data/bloom_task_grid/Exp23_Representation_Wu_Lim.ipynb) | BLOOM 7b1 | Do representation-only measures (Semantic Hub, Lim et al. style) agree with patching? | Largely yes: high similarity for IOI and translation, lower for word order. | Single model | 1.4.4 |
| [`Exp24_Concept_Gender_Feature_vs_Lookup`](natural_data/direction_transfer/Exp24_Concept_Gender_Feature_vs_Lookup.ipynb) | Gemma-2 2B–27B-it; FR, ES, AR, HI | Is gender a language-agnostic feature or a shared concept plus per-language lookup? | Gate passed: the gender concept is extractable in all models and languages. HI/AR item banks are not native-speaker verified. | Gate / setup | 1.5.1 |
| [`Exp25_WordOrder_Parallel`](natural_data/direction_transfer/Exp25_WordOrder_Parallel.ipynb) | BLOOM 1b7 / 7b1; 7 languages, parallel scenes | Is word-order computation shared when the items are parallel across languages? | Clause-order flip fraction ≈ 0 for SVO recipients; CKA high among European languages (0.95–1.00) yet computation separated. Confirms Exp22 with parallel items. | Two models | 1.5.2 |
| [`Exp27_A_WordOrder_Gate`](natural_data/direction_transfer/Exp27_A_WordOrder_Gate.ipynb) | Gemma-2 9B | Does word-order state transfer under whole-state patching? | **Failed instrument, not a result.** Verdict INVERSION_ABANDONED: same-language ceiling controls invert in 10 of 15 cases (1 transfer, 10 wall, 4 ambiguous). | Kept for the record | 1.5.3 |
| [`Exp28_B_Gender_DTI`](natural_data/direction_transfer/Exp28_B_Gender_DTI.ipynb) | Gemma-2 9B | Same instrument applied to grammatical gender | **Failed instrument, not a result.** Verdict INSTRUMENT_INADEQUATE: only 2 usable recipient languages (3 needed); HI fails the accuracy gate. | Kept for the record | 1.5.4 |
| [`Exp29_DirectionTransfer_DAS`](natural_data/direction_transfer/Exp29_DirectionTransfer_DAS.ipynb) | Gemma-2-9B-it | Does an additive direction for gender / word order transfer across languages, specifically? | Gender: 11 of 12 cross-language cells transfer. Word order: 6 of 7 walled (DE→EN ambiguous). Each direction is specific to its own feature. | **One exploratory run, one instruction-tuned model**; pre-registered; HI/AR items not native-verified | 1.5.5 |

### Part 2: synthetic models

| Notebook | Model / substrate | Question | Key result | Seeds / evidence | Report § |
|---|---|---|---|---|---|
| [`Exp01_Capacity_Competition`](synthetic/allocation_toy/Exp01_Capacity_Competition.ipynb) | Closed form (no training) | Does the allocation rule produce a shared → specialised transition? | Yes, with heterogeneity; the symmetric case jumps in one step. Both comparative statics hold. | Analytic | 2.1 |
| [`Exp02_Learnable_Allocation`](synthetic/allocation_toy/Exp02_Learnable_Allocation.ipynb) | Small trained network with shared and private routes | Does SGD peel languages off to private routes in the predicted order? | Order matches the closed form; timing is about 2× earlier than the formula. Three bugs found and fixed along the way. | Partial confirmation; seed count not stated in the report | 2.2.1 |
| [`Exp02b_Anchor_Bias`](synthetic/allocation_toy/Exp02b_Anchor_Bias.ipynb) | Same family, 4 languages, 70% anchor sampling | Does imbalanced training alone create a cost for the minority languages? | Gap under biased sampling (anchor ≈ 0.054 vs 0.12–0.16 for minorities at width 450); shrinks slowly from width 1600 to 4800. Suggestive, not conclusive. | Several widths | 2.2.3 |
| [`Exp03_Number_Agreement_PCFG`](synthetic/pcfg_agreement/Exp03_Number_Agreement_PCFG.ipynb) | PCFG number agreement | Do a cheaper shortcut and a language conflict shift the capacity needed? | Shortcut manipulation works (acc 0.999 vs 0.958). The conflict manipulation partly engages but never shifts the capacity at which the task is solved; four designs were bypassed by the model. | 5-seed replication on one comparison | 2.3 |
| [`Exp03a_Weight4_Extended`](synthetic/pcfg_agreement/Exp03a_Weight4_Extended.ipynb) | PCFG number agreement | Does reweighting the rare override (weight 4) give a usable instrument? | Follow-up on the Exp 3 instrument; the report lists it among Exp 3 related notebooks without a separate headline result. | Partly run (2 of 10 code cells with output) | 2.3 |
| [`Exp04_Conflicting_Gender_Followups`](synthetic/conflicting_gender/Exp04_Conflicting_Gender_Followups.ipynb) | Tiny network, shared nouns with conflicting genders | Does conflict cost capacity, and does it grow with overlap? | Conflict costs capacity (solved at width 5 vs 4 without), but the required width is flat across overlap 0–0.75: a switch, not a gradient. The disjoint-vocabulary control was never run. | Single sweep | 2.4 |
| [`Exp05_AllenZhu_Paired_CFGs`](synthetic/paired_cfgs/Exp05_AllenZhu_Paired_CFGs.ipynb) | Depth-7 paired CFGs, GPT-2-style | Does sharing terminals across two grammars cost capacity? | **Designed, never run.** Grammar and parser checks pass; the 60-run sweep (about 100–150 GPU-hours) was not started. | No results | 2.5 |
| [`Exp06_SharedTree_DivergentLinearization`](synthetic/shared_tree_linearization/Exp06_SharedTree_DivergentLinearization.ipynb) | One abstract tree, 3 word orders (L_A, L_B partial reversal, L_C full reversal) | Does the shared model pay for differing word order? | At small capacity the shared model pays extra loss ordered by reordering distance (C > B > A). At d=96: 4-head runs show none through 180K steps, 12-head runs show it persisting (2 seeds). Unresolved. | 2 seeds; configuration-dependent | 2.6.1 |
| [`Exp06_ConvergenceCheck_d96`](synthetic/shared_tree_linearization/Exp06_ConvergenceCheck_d96.ipynb) | Same grammar, 4-head, d=96 | Is the "no extra loss" at d=96 real convergence or under-training? | Extended to 180K steps: no extra loss for the 4-head configuration. | 2 seeds | 2.6.1 |
| [`Exp06b_Syncretism_BudgetMatched`](synthetic/shared_tree_linearization/Exp06b_Syncretism_BudgetMatched.ipynb) | Shared tree + terminal collapse | Can collapsing terminals create a cost for sharing? | Local context resolves most of the collapse; design abandoned because it tests input destruction rather than the intended effect. | Abandoned design | 2.6.4 |
| [`Exp07a_DualRole_tAgree`](synthetic/shared_tree_linearization/Exp07a_DualRole_tAgree.ipynb) | Shared-tree PCFG, one terminal with two roles | Does a dual-role terminal make the shared model lose? | No: shared matches or beats dedicated at every learnable capacity. Null control. | Seed count not stated in the report | 2.7.1 |
| [`Exp07b_Serialization_Sharing_v1`](synthetic/shared_tree_linearization/Exp07b_Serialization_Sharing_v1.ipynb) | Shared tree, 3 serializations; interchange tests | Does the model share machinery across serializations of the same tree? | Near-floor sharing at tree levels where word order diverges, near-ceiling where it does not. **The report's claim rests on this version.** | Evaluated on 3 trained checkpoints (`bp_h4_d96_s0`, `s7`, `s13`), separately, no pooled statistics; thresholds never frozen, no verdict; E4 calibration not run (`RUN_E4_LAUNCH` False) | 2.7.2 |
| [`Exp07b_Serialization_Sharing_v2`](synthetic/shared_tree_linearization/Exp07b_Serialization_Sharing_v2.ipynb) | Same | Same question, revised instrument | Kept beside v1. Its header says v1's first run could not tell right-symbol from wrong-symbol patches at levels 4–5, which motivated the change. The report's claim rests on v1, not this version. | Same 3 checkpoints, separately, no pooled statistics; thresholds never frozen, no verdict; instrument revised after seeing results (v1's numbers); the code-swap measurement (E1b) returns no usable numbers and the E2 profile control fails; E4 launch and later cells errored | 2.7.2 |
| [`Exp08a_Three_Checks`](synthetic/conversion_cost/Exp08a_Three_Checks.ipynb) | Shared tree, translators and joint models, d=8/16/32 | Does the cost of converting a rendering predict how well it shares? | No. Untestable at d=16 (gate fails); at d=8 Spearman −0.13, and a composition does not share better than its parts. Conversion cost itself is stable (0.77 across d). | 3 seeds at d=8 only | 2.8 |
| [`Exp08b_Valid_Capacity_d12`](synthetic/conversion_cost/Exp08b_Valid_Capacity_d12.ipynb) | Same, d=12 | Is there a capacity where both translators and the sharing gate work? | Follow-up to the three checks; the report discusses it with them and gives no separate headline number. | Partly run; uses `exp8_lib` (see Environment notes) | 2.8 |
| [`Exp08c_Conversion_Cost_vs_Eps`](synthetic/conversion_cost/Exp08c_Conversion_Cost_vs_Eps.ipynb) | Exp06 grammar, d=96 | Does translator cost order the shared-tree gap? | Fails: the fully reversed language is cheap to convert (≈ 0) but costs more to share than the partially reversed one (0.045–0.063). | 2 seeds in saved output | 2.6.2 |
| [`Exp08d_Transfer_Test`](synthetic/conversion_cost/Exp08d_Transfer_Test.ipynb) | Exp06 grammar, d=96 | Does low conversion cost mean faster adaptation? | Fails: the reverse ordering held in all 3 seeds (frozen prediction not met). | 3 seeds | 2.6.3 |
| [`Exp09_Prelude_Factorial_Exploratory`](synthetic/hardened_substrate/Exp09_Prelude_Factorial_Exploratory.ipynb) | Exp06/07 substrate | Exploratory grid over overlap, data and capacity | Exploratory dump, no claim. Similarity noise floor is high (different-seed 0.723); interchange retention anchor→L_B 0.969, anchor→L_C 0.707. | Exploratory; two cells errored | 2.8 (factorial) |
| [`Exp09_Hardened_Substrate_Sanity`](synthetic/hardened_substrate/Exp09_Hardened_Substrate_Sanity.ipynb) | Harder PCFG, d=96 | Does the harder grammar put a monolingual model in the capacity-constrained regime? | Yes: accuracy 0.554 (target band 0.5–0.85). The saved run is degree 4 although the filename says Degree16. | Sanity check, single run | 2.9 |
| [`Exp09b_OptionB_Smoke`](synthetic/hardened_substrate/Exp09b_OptionB_Smoke.ipynb) | Harder PCFG; two languages with the same word order, differing only in a context-keyed terminal lookup | Does the shared model pay when the conflict is a real terminal lookup rather than word order? | **Single-seed smoke test.** The shared model is worse than the dedicated one at conflicted tokens (CE gap +0.044 to +0.087 at depth 1; larger at depth 2, e.g. +0.151 at d=8). Promising, not replicated. | **Single-seed smoke test** (`seeds=[0]`, 9 models) | 2.9 |

## Environment notes

- **Models:** Gemma-2 (2B and 9B base, 9B-it, 27B-it) and BLOOM (1b7, 7b1) for Part 1. Part 2 uses small networks trained from scratch (closed form for Exp01; small networks for Exp02-04; rotary GPT-style transformers for Exp05-09).
- **Hardware and paths:** notebooks were run on a shared GPU server. They hard-code GPU ids (`CUDA_VISIBLE_DEVICES`, `cuda:N`), local model paths such as `/home/models/gemma-2-9b-it`, and a home-directory path in saved tracebacks. Several Part 1 notebooks also set an institutional HTTP proxy. These were left unchanged.
- **Saved outputs, not reruns:** every notebook is provided with its saved outputs and is not meant to be rerun as-is. Several expect files that are not in the repo (cached results folders, checkpoints, `./exp8/` training outputs).
- **`exp8_lib`:** not a separate file. It is written by cell 2 (`%%writefile exp8_lib.py`) of [`archive/synthetic/conversion_cost/Exp08_Adapter_Cost.ipynb`](archive/synthetic/conversion_cost/Exp08_Adapter_Cost.ipynb). That cell holds the whole library (grammar, the ten renderings, joint and translator training, `TinyRotaryGPT`) and covers every name `Exp08a` and `Exp08b` use. Order: run only that cell to create `exp8_lib.py` next to the notebooks; then `Exp08a_Three_Checks`; then `Exp08b_Valid_Capacity_d12`. Both read training outputs under `./exp8/` that are not stored in the repo. `Exp08c` and `Exp08d` define their own model code and do not use it (run `Exp08c` before `Exp08d`, which reads its results).
- **Hugging Face token:** six notebooks used to contain a hard-coded token. It has been replaced with `os.environ.get("HF_TOKEN", "<REDACTED>")`; set `HF_TOKEN` in your environment to run them.
- **Packages:** `requirements.txt` lists the packages the notebooks import, unpinned (versions were not recorded).
- **Notation:** Exp01-Exp09 use notation from an earlier theoretical framing that is no longer being pursued.

## Known limitations

- Exp07b (interchange tests) was evaluated on 3 trained checkpoints, separately, with no pooled statistics; thresholds were never frozen and there is no verdict. v2's instrument was revised after seeing results, its code-swap measurement (E1b) returns no usable numbers, and its E2 control fails. The report's claim rests on v1.
- Exp09b (Option B smoke test) is a single-seed run (`seeds=[0]`).
- Exp29 is one exploratory run on Gemma-2-9B-it. Whether it holds on base models or other sizes is untested.
- Exp27 and Exp28 are kept as failed-instrument notebooks, not as results.
- The HI and AR item banks in Exp24 and Exp29 are not verified by native speakers.
- Exp05 (paired grammars) was designed and validated but never run.
- At d=96 the toy word-order cost depends on the architecture (4 vs 12 attention heads); the reason is unknown.
- Distant-language comparisons (AR, SW, HI, ZH) are partly confounded by tokenizer issues; the strongest results are on European languages plus YO.
- Several kept notebooks contain saved error cells from interrupted or partial runs; the reports note where a result is partial.

## Provenance

Files were renamed and regrouped with `git mv`; `docs/RENAMES.md` maps every old name to its new path and `git log --follow` works on each. The only edit inside any notebook is the token replacement above.
