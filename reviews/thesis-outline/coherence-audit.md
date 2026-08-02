# Cross-Document Coherence Audit: Thesis Outline

Audited at snapshot commit 353c4dd (branch `thesis-outline`). Scope: whole-document
content coherence, RQ threading, open items, page budget, and a simulated quick review.
Sentence style was out of scope. No project file was edited; this report is the only
deliverable.

## Executive summary

1. The document compiles clean at the stated baseline: 0 errors, 0 undefined references, 0 undefined citations, no duplicate labels, 105 PDF pages.
2. The story is coherent end to end: problem, gap, five RQs, three method families, and the evidence design all reuse one formalization, and every RQ threads through all six chapters at the label level.
3. The single biggest supervisor decision is the summarization family: it is promised in RQ2, RQ3, RQ4 and hypothesis H1, yet zero summarization configurations are trained or measured, and the sources themselves say the family comes out of the RQs if the runs do not happen.
4. The second decision is the RQ4 matching instrument: random within-type merging is not built, so RQ4 is currently answerable only for the partitioning and sparsification pairs, and the spanning-forest pairing needs a dropout-rate decision (0.3 versus 0.419) that invalidates existing runs either way.
5. The Contributions section of Chapter 1 is empty: its entire content sits commented out in the source, so the compiled introduction makes no contribution claim at all.
6. The page budget is far exceeded: 88 countable two-column pages against a limit of 35, of which 46 pages are the unfiled results gallery; even without the gallery the count is 42.
7. The Preliminaries meet their stated 4-page budget exactly (pages 5 to 8), but the Data section alone occupies 13 pages (19 to 31), which is the largest prose overrun.
8. Three promised measurements gate the headline claims: the H1 receptive-field metric (specified, not implemented), the RQ5 lossless positive control (not trained), and the RQ1a random and recipe split runs (not made).
9. RQ1 as stated promises a standard-encoder comparison group that the methodology only conditionally commits to ("may be added if time permits"), and RQ1a has no answer slot in the Conclusion.
10. What is on track: the compile hygiene, the RQ threading skeleton, the measured RQ1/RQ2/RQ3 partitioning and sparsification evidence, the honest FABRICATED watermarking of placeholder floats, and the Discussion limitations prose, which is already near-final quality.

## 1. Compile and label audit

Recipe run as specified (pdflatex, bibtex, pdflatex twice) from a clean checkout of
353c4dd.

| Check | Result |
|---|---|
| LaTeX errors (`^!` in log) | 0 |
| Undefined references | 0 |
| Undefined citations | 0 |
| Duplicate labels across `sections/` | none |
| Multiply-defined label warnings | none |
| Output | msc_thesis.pdf, 105 pages (baseline matched) |
| Overfull boxes | 130 (cosmetic; expected in two-column with wide tables) |

Every `\ref` and `\cite` target resolves. The only remaining warning is the standard
"Label(s) may have changed" from the final rerun ordering, which a fourth pass clears.

## 2. RQ threading matrix

One row per research question. A cell holds the section label that carries the thread, or
MISSING. "(stub)" marks a heading that exists with empty or comment-only content, which is
acceptable at outline stage but listed for completeness.

| RQ | Introduction (stated) | Prelim/Related Work (grounded) | Methodology (answerable) | Results (planned exhibit) | Discussion (interpreted) | Conclusion (answered) |
|---|---|---|---|---|---|---|
| RQ1 | `sec:intro:rq1` | `sec:prelim:formalization`, `sec:relwork:ml4eda:qor`, `sec:relwork:ml4eda:baselines` | `sec:method:architecture:baselines`, Phase 1 of `sec:method:experiment:design` | `sec:results:rq1` (+ gallery `sec:results:gallery:rq1`) | `sec:discussion:interpretation:feasibility` (stub), `sec:discussion:relatedwork:prediction` (stub) | `sec:conclusion:rq1` (stub) |
| RQ1a | `sec:intro:rq1` (paragraph) | `sec:relwork:ml4eda:protocols` | `sec:method:experiment:splitting`, `tab:split_protocols` | `sec:results:rq1:protocol` (+ gallery `sec:results:gallery:rq1a`) | `sec:discussion:limitations:validity` | MISSING (no RQ1a subsection or mention in Chapter 6) |
| RQ2 | `sec:intro:rq2` | `sec:prelim:reduction:measuring`, `sec:prelim:gnn:cost`, `sec:relwork:scaling` | `sec:method:reduction`, Phase 2, `sec:method:experiment:metrics:efficiency`, `sec:method:experiment:metrics:reduction` | `sec:results:rq2` (+ gallery `sec:results:gallery:rq2`) | `sec:discussion:interpretation:tradeoff` (stub) | `sec:conclusion:rq2` (stub) |
| RQ3 | `sec:intro:rq3` (+ H1) | `sec:prelim:gnn:cost` (over-squashing), `sec:relwork:reduction:forgnn` | Phase 3, `sec:method:experiment:metrics:accuracy`, positive control in `sec:method:experiment:design` | `sec:results:rq3` (+ gallery `sec:results:gallery:rq3`) | `sec:discussion:interpretation:tradeoff` (stub) | `sec:conclusion:rq3` (stub) |
| RQ4 | `sec:intro:rq4` (+ H2) | `sec:relwork:domain`, `sec:relwork:gap`, `sec:prelim:reduction:measuring` | `sec:method:summary:random` (matching instrument, NOT BUILT), pairing design in `sec:results:rq4:pairings` comments | `sec:results:rq4` (+ gallery `sec:results:gallery:rq4`) | `sec:discussion:interpretation:domain` (stub), `sec:discussion:limitations:methods` | `sec:conclusion:rq4` (stub) |
| RQ5 | `sec:intro:rq5` | `sec:relwork:generalization:shift`, `sec:relwork:generalization:mismatch` | two-pass protocol in `sec:method:experiment:metrics` (eqs. matched/full), `sec:method:summary:schema`, `sec:method:architecture:exact` | `sec:results:rq5` (+ gallery `sec:results:gallery:rq5`) | `sec:discussion:interpretation:generalization` (stub) | `sec:conclusion:rq5` (stub) |

Reading of the matrix: the skeleton is complete except for one hole, the missing RQ1a
answer slot in the Conclusion. Every Conclusion cell is a labelled stub blocked on
Chapter 4, which is expected at this stage; the mapping comment at
`sec:conclusion:answers` already pairs each RQ with its qualifying limitation, so the
intended final column is fully designed even though unwritten.

Three threads are labelled but not yet backed by runnable or run evidence:

- RQ1: the RQ text names three comparison groups, but the standard-encoder group (GCN, GraphSAGE, GIN) is only conditional in `sec:method:architecture:baselines` ("may be added if time permits"), and the gallery caption for `fig:rq1_baseline_tiers` records that the standard-encoder and DeepGate4/HOGA rows are invented placeholders. Only SynthNet has run, and it collapsed to a constant.
- RQ4: the random within-type merging control does not exist, and neither domain-specific summarization candidate (cone versus MFFC) has been chosen or measured.
- RQ5: the exact colour-refinement positive control has not been trained (`fig:rq5_positive_control` is ENTIRELY FABRICATED), and the CPU inference half of the claim has not been run.

## 3. Consolidated open-items inventory

Sources: every `.tex` under `sections/`, plus `writing_plan.md` (the authoritative
intended-content map). Grouped by what unblocks each item.

### (a) Needs a measurement or run

Training or evaluation runs:

1. RQ1a random-split and recipe-split training runs at the headline configuration (`sections/methodology/experiment-setup.tex:123`, `sections/4-results.tex:46`; writing plan marks them IN FLIGHT).
2. The entire summarization family: no method trained or measured (`sections/4-results.tex:222`, gallery subsection `sec:results:gallery:summarization`).
3. Random within-type merging: not built; cost is one training run per matched point (`sections/methodology/reduction-summarization.tex:180`).
4. RQ5 positive control: colour-refinement configuration end to end (`sections/4-results.tex:761`).
5. CPU inference pass for RQ5's modest-hardware claim (`sections/4-results.tex:751`).
6. RQ4 seed variance: either seeds on the RQ4 pairs or point-estimate framing (`sections/4-results.tex:681`, `sections/methodology/experiment-metrics.tex:213`).
7. Random edge dropout at rate 0.419 if the spanning-forest pairing is to be matched (`sections/methodology/reduction-sparsification.tex:32`, `sections/4-results.tex:150`).
8. Standard-encoder baseline runs (GCN, GraphSAGE, GIN), if RQ1 keeps that clause (`sections/methodology/baselines.tex:6`).

Offline measurements and probes:

9. H1 receptive-field metric: specified at `eq:method:receptive`, not implemented (`sections/methodology/experiment-metrics.tex:190`, `sections/introduction/research-questions.tex:48`).
10. Residual-redundancy probe, colour refinement at depths 1 to 4 (`sections/methodology/reduction-summarization.tex:134`).
11. Cone and MFFC compression measured on the real corpus, one shard suffices (`sections/methodology/reduction-summarization.tex:103`).
12. ConvMatch per-graph probe-sampling cost (`sections/methodology/reduction-summarization.tex:156`).
13. Whole-corpus tier statistics: regenerate `media/tables/corpus_tiers.tex` from the cluster CSV; the compiled document currently shows a visible NOT YET GENERATED placeholder (`sections/methodology/data.tex:312,501`).
14. Tier count confirmation (924,220 labelled, ~874,220 after holdout) before the numbers are stated (`sections/methodology/data.tex:203`).
15. Tier-0 distinctness count after structural hashing (`sections/methodology/data.tex:313`).
16. Immediate-dominator degeneracy on this corpus, asserted not measured (`sections/related-work/prelim-aig.tex:105`).
17. Paired memory savings recomputed on marginal memory rather than raw peak (`sections/methodology/experiment-metrics.tex:142`).
18. GPU-hours and node-hours consumed, for the ethics paragraph (`sections/5-discussion.tex:192`).
19. Step-21 choice-network node-count understatement (43 of 200 cases): decide where it is reported (`sections/methodology/data.tex:184`).
20. Generator additions: summarization rows, RQ1 baseline rows, consolidated summary table; and its cross-state caption says RQ4 where this document says RQ5 (`sections/4-results.tex:6`).

### (b) Needs an author decision

1. Keep or cut the summarization family from the RQs and methodology, depending on whether its runs will happen. Everything in group (a) items 2, 3, 4, 10, 11 hangs on this.
2. Which domain-specific summarization candidate runs: cone or MFFC (`sections/methodology/reduction-summarization.tex:68`).
3. Dropout rate 0.3 versus 0.419: changing it invalidates existing random-dropout runs; not changing it leaves the spanning-forest pair unmatched.
4. H1: build the receptive-field metric or downgrade H1 to a discussion point.
5. RQ1 scope: keep or drop the standard-encoder comparison group from the RQ text.
6. Restore or delete the commented-out Contributions prose; as compiled, Chapter 1 claims no contributions (`sections/introduction/contributions.tex`).
7. Add an RQ1a answer slot to the Conclusion, or state that RQ1a folds into RQ1's answer.
8. File the 46-page gallery: each float moves into its section, to the appendix, or is deleted (`sec:results:gallery`).
9. Title, student number, and thesis start date placeholders in `msc_thesis.tex` (TODO markers at lines 25 to 39).
10. Abstract: currently the single word "Placeholder"; the skeleton exists as a comment.
11. Code and data availability: hosting, license, what ships (`sections/methodology/experiment-reproducibility.tex:23`).
12. Where the variance-decomposition and stratified findings live: `fig:rq1_variance_decomposition` (structural residual R^2 of 0.124 against the headline 0.343) and the tier/source stratifications are load-bearing measured results currently owned by no RQ section.
13. Four missing citation keys: LSOformer and Jiang et al. (`relwork-ml4eda.tex:36`), a script-generation reference (`:53`), OpenLS-DGF and the ML4EDA position paper (`:95`), a distributed-GNN reference (`relwork-scaling.tex:23`).
14. Level-over-RWSE structural argument, one sentence, if one exists (writing plan, architecture).

### (c) Cosmetic or bookkeeping

1. `%TODO intro sentence` for the Data section (`sections/methodology/data.tex:4`).
2. Pin the abc version in the appendix (`sections/7-appendix.tex:68`) and library versions in hardware (`experiment-setup.tex:219`); also the wall-clock budget.
3. Record the HP search space and trial count (`experiment-setup.tex:184`).
4. Confirm Orchestrate per-pass visit order (`prelim-algorithms.tex:61`) and the GNN+ per-message activation against the reference implementation (writing plan, architecture).
5. Confirm the positional-encoding min-pooling adaptation before defending it; note the typo "positinal" in that comment (`architecture.tex:212`).
6. Search Snellius home and scratch for the synthetic-generator script and seed (`data.tex:140`).
7. Fix `data/DATA_README.md` and `AIG_DATASET_README.md` to agree with the thesis on source provenance (writing plan, Source Circuits; repo-side, not thesis-side).
8. Numerical determinism subsubsection is a one-line TODO; the writing plan holds its full outline (`experiment-reproducibility.tex:19`).
9. The `<30%` recipe-overlap statistic was dropped from `sec:prelim:synthesis:scripts` (the prose now argues order-dependence without the number); the writing plan's TODO(source) is thereby resolved by omission. Fine, but note the motivating statistic is gone.

### Writing-plan entries with no corresponding realization in the .tex

The writing plan is otherwise faithfully realized; these are the gaps:

1. Contributions itemized list (five items including RQ1a as a named contribution): present in the plan and in the source only as a comment block; no compiled prose.
2. Reproducibility subsubsections `determinism` and `availability`: full outlines in the plan, empty one-line-TODO stubs in the tex.
3. Discussion subsections 5.1.2 (`sec:discussion:relatedwork:reduction`), 5.1.3 (`sec:discussion:relatedwork:incomparable`), 5.3 (`sec:discussion:implications`), and 5.5 (`sec:discussion:outlook`): heading-only stubs; the plan carries guidance for implications and outlook but the reduction-positioning and incomparability subsections have neither plan guidance nor tex content.
4. Appendix sections A.2 through A.5: comment-only stubs whose intended tables are specified in the plan but not started.
5. The plan's consolidated-summary note says `analysis/tables.py::summary` now generates the table `sec:results:summary` promises; the section itself is still empty and the generated table is only `\input` in the gallery, not in the section that promises it.

## 4. Page-budget estimate

Limit: 35 two-column pages counted from the Introduction, excluding references and
appendix. Page numbers from the compiled table of contents.

| Chapter | Pages | Count | Note |
|---|---|---|---|
| 1 Introduction | 1 to 4 | 4 | on budget |
| 2 Preliminaries & Related Work | 5 to 12 | 8 | Preliminaries 5 to 8 (4 pp, exactly the stated ~4-page budget); Related Work 9 to 12 (4 pp) |
| 3 Methodology | 13 to 36 | 24 | Architecture ~3.5, Reduction ~3, Data 19 to 31 (13 pp), Experiments 32 to 36 (5 pp) |
| 4 Results | 37 to 84 | 48 | prose sections 4.1 to 4.6 are 2 pp; the unfiled gallery 4.7 is 39 to 84 (46 pp) |
| 5 Discussion | 85 to 86 | 2 | mostly Limitations and Ethics prose |
| 6 Conclusion | 87 to 88 | 2 | stubs |
| Countable total | 1 to 88 | 88 | against a limit of 35 |
| Bibliography | 89 to 92 | 4 | excluded from the count |
| Appendix | 93 | 1 | excluded from the count |

Reading: the raw overrun (88 versus 35) overstates the problem because 46 pages are the
gallery, which is explicitly a holding pen. Excluding it leaves 42 pages, still 7 over,
before the Results prose, the Discussion, and the Conclusion are actually written. The
chapters far over any plausible implied budget are:

- Data (13 pages): the largest prose block in the document. The source-circuits narrative, the generation mechanics, and the dataset-analysis figures each carry appendix candidates. A 35-page thesis cannot spend more than a third of its methodology share on data.
- Results (48 pages nominal): filing the gallery is the single largest lever. Each kept float costs roughly half a page; the current gallery holds about 45 floats plus a dozen generated tables, so keeping even half of them consumes the entire Results budget.
- Methodology overall (24 pages): against a plausible 10-to-12-page share of a 35-page thesis, roughly half must move to the appendix or compress, with Data the first target.

Preliminaries are the one place the budget discipline already worked: the stated ~4-page
budget is met, though the source still carries the "20 subsubsections against ~4 pages"
TODO, which can now be retired or revalidated after the merge candidates were folded.

## 5. Whole-document review pass (simulated quick assessment)

Run as a content-level quick assessment (style out of scope). Findings:

Coherent: yes, end to end. The problem-gap-RQ chain is explicit, and
`sec:prelim:formalization` restates all five RQs as statements about the same four
objects, which makes the document read as one study rather than five topics. The gap
claim is properly narrowed after CTS-Bench into a five-part enumerated claim. Three
design elements a committee will credit: the provably lossless positive control, the
random-merging matching instrument, and the three-protocol leakage measurement.

Do the planned experiments answer the RQs as stated? Mostly, with four exceptions,
all already listed above: the RQ1 standard-encoder clause versus the conditional
methodology; the unbuilt RQ4 matching instrument; the unimplemented H1 metric; and the
untrained RQ5 positive control.

Orphans: RQ1a has no Conclusion slot. The variance-decomposition and stratified-scoring
results are strong measured evidence not owned by any RQ section. The spanner negative
result is properly attached (related work plus sparsification section) and is not an
orphan.

The panel's overall note: the outline is internally consistent about its own gaps, with
placeholder data loudly watermarked, which is to its credit. The risk is not incoherence
but breadth: the promised evidence set (summarization family, H1 metric, positive
control, protocol runs, seed variance) exceeds what the remaining time plausibly allows,
and the document already contains the honest fallback for each (drop the family, downgrade
H1, report point estimates). The supervisor conversation should decide which fallbacks to
take now rather than at write-up.

## 6. CLAUDE.md thesis-writing conformance (content-structure level, noted not fixed)

Checked against the "Thesis writing" section of the repository CLAUDE.md. Not style
policing; only systematic, structural observations.

1. Commented-out prose left in the file: `sections/introduction/contributions.tex` holds its entire content (an itemized five-part contribution list plus a closing paragraph) as a comment block. The rules say commented-out prose is restored or deleted; the scaffolding exception covers stubs, but this is finished prose that also exists in `writing_plan.md`, so it is duplicated in exactly the way the plan file was created to avoid.
2. Multi-line reasoning comments with `\ref` chains survive in several files despite the one-line rule, most heavily in `4-results.tex` (99 comment lines, including the multi-line holding-pen preamble), `5-discussion.tex` (35), and `6-conclusion.tex` (25). Most are legitimate outline scaffolding under the stated exception; the ones that read as self-addressed reasoning rather than scaffolding are the RQ4 pairing rationale in `4-results.tex` and the positive-control note commented out at `research-questions.tex:81`.
3. Second-person instruction register inside comments, which the rules name as the tell: "do not write as though it is" (`reduction-summarization.tex:64`), "do not report it as evidenced" (`experiment-metrics.tex:190` and the caption of `fig:h1_receptive_field`).
4. Structure announcements in prose: "This section describes the reduction methods evaluated in this thesis." (`reduction-partitioning.tex:3`) and "This subsection characterises the corpus..." (`data.tex`, Dataset Analysis opening) are the two clear cases; both open with the announcement rather than the claim.
5. Em and en dashes: none found in any non-comment prose line. The rule is being followed.
6. Fabricated placeholders are uniformly and loudly marked (FABRICATED watermarks, TODO tags, a visible NOT YET GENERATED table placeholder), which conforms to the honesty conventions the CLAUDE.md and writing plan establish.
7. One typo in a comment worth catching before submission since comments ship with the source: "positinal" (`architecture.tex:212`).

## Suggested supervisor-meeting agenda (derived from the above)

1. Go/no-go on the summarization family (drives items a.2-4, a.10-11, b.1-2).
2. H1: implement the metric or downgrade the hypothesis.
3. RQ1 baseline scope: commit to or drop the standard-encoder tier.
4. Dropout-rate decision for the spanning-forest pairing.
5. Seed policy for RQ4 (seeds on pairs versus point estimates).
6. Page-budget plan: file the gallery, compress Data, appendix policy.
7. Restore the Contributions prose and add the RQ1a conclusion slot.
8. Housekeeping: abstract, title decision, metadata placeholders, missing citation keys.
