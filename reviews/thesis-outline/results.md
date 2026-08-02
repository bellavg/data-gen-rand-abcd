# Results chapter, outline content review

Unit: `IV_Gardner___Master_AI_Thesis_Outline/sections/4-results.tex`.
Baseline: snapshot commit 353c4dd. Compile after edits: 0 errors, 0 undefined
references, 0 undefined citations, 105 pages. `chktex` on the file: the same two
warnings as before the edits (interword spacing, non-breaking space), no new ones.

## Findings, ranked

### Fixed in this pass

1. **Three of the six research questions had no named exhibit.** RQ1a, RQ4 and
   RQ5 stated what would be compared but pointed at no figure or table, so a
   reader could not tell whether the evidence existed. Every subsection now
   carries a one-line stub naming the exhibit that answers it, and every
   generated table now has a commented `\input` at the section that owns it.
   The chapter can be read as an evidentiary plan rather than a list of
   intentions.
2. **The gallery was an undispositioned holding pen.** All 47 figures and 15
   generated tables now carry a one-line comment saying where each goes: its own
   section, the appendix, or deletion. Full list below. No prose was written
   around any float and no number was typed into a generated table.
3. **The chapter header carried two stale claims.** It said the generator lacks
   the consolidated summary table and that the cross-state caption is mislabelled
   RQ4. Both are now false: `summary_all.tex` is generated, and the current
   generator labels the cross-state table RQ5. The header now points at the
   fabricated-run inventory in the gallery header instead of duplicating it.
4. **The summary section promised a table it never pointed at.** The promise is
   kept rather than removed, since the generator now builds it. The section
   carries a commented `\input` for `tab:summary_all`.
5. **Two code identifiers in outline prose.** The throughput subsection named two
   logged metric keys in typewriter font. Replaced with the quantities they
   measure (time per optimisation step, time per epoch). The remaining
   identifiers sit in figure captions and are flagged below rather than edited,
   because editing them means rewriting the author's sentences.
6. **The path macros were declared mid-file, after the commented `\input` lines
   that use them.** Moved to the top of the chapter. No new declaration was
   added. See the macro note below.
7. **The gallery header inventory was missing two outstanding items.** The
   refinement-depth probe and the whole-corpus statistics were absent from the
   list the chapter header now delegates to, while floats for both sit in the
   holding pen. Both added.

### Needs the author

Ranked by how much of the chapter each one blocks.

1. **The summarization family has not been run at all.** No method in it has been
   trained or measured, and four gallery floats are entirely invented. This is not
   one missing row. It takes out the third pairing kind of RQ4, the RQ5 positive
   control, the RQ3 validity gate (the exact configuration must land on the
   full-graph baseline, and a deviation invalidates the rest of the chapter), and
   the lossless-coarsening contribution claimed in the introduction. Either the
   runs land or the family comes out of the methodology and the research questions
   too, not only out of the results.
2. **Every configuration is trained exactly once.** RQ4 asks whether domain
   knowledge buys anything, expects a small effect, and has no run-to-run variance
   to judge a gap against. The training-trajectory figure shows validation
   explained variance swinging by several units between consecutive epochs on some
   configurations, which is larger than the effect RQ4 is looking for. Seeds on
   the RQ4 pairs only, or the differences reported explicitly as point estimates
   that cannot be separated from noise. There is no third option that is honest.
3. **One of the four RQ4 pairings is not matched.** Spanning forest retains 58.1
   percent of edges against random edge dropout's 69.7 percent at the configured
   rate. Approximately 0.419 matches them. Until that run exists the pair cannot
   be reported as matched, which leaves RQ4 with two clean pairings by
   construction and one by calibration.
4. **RQ1a needs two training runs.** Only the design-disjoint protocol has been
   run. Without the random and recipe-disjoint rows the inflation factor cannot be
   computed, which also blocks the translation the discussion chapter plans to use
   when comparing against published numbers obtained under leakier protocols.
5. **RQ1's baseline tiers 2 and 3 are unrun.** The standard encoders under
   identical training are invented, and of the published models only SynthNet ran,
   collapsing to a constant prediction. So "accurately" currently has a referent
   only against the trivial predictors, and the size-only regressor outranks the
   encoder on rank correlation. That is a reportable finding, but the chapter
   should say which of the three tiers it is actually claiming to beat.
6. **The receptive-field metric is specified and not implemented.** H1 is
   asserted, not tested. Two gallery floats are placeholders for it. Either build
   it or downgrade H1 to a discussion point, per the note already in the
   methodology.
7. **The processor-side inference pass has not been run.** Every surviving
   inference record is from the accelerator, so RQ5's practical claim, that a model
   trained cheaply on reduced graphs can serve full graphs on modest hardware, has
   no measurement behind it.
8. **Smooth L1 loss is promised twice and delivered nowhere.** The RQ1 accuracy
   subsection and the RQ5 accuracy subsection both name it, and no generated table
   reports it. It is a training loss, not an evaluation metric. Either drop it
   from both promises or add it to the generator.
9. **Three captions carry material that should not appear in the document.** The
   baseline-group caption and the processor-inference caption name source files
   and a device string in typewriter font, which the abstraction-level rule keeps
   out of prose. The variance-decomposition caption ends by telling a section what
   number it should defend, which reads as a note to self rather than an author's
   note. These are the author's sentences, so they were flagged rather than
   rewritten.
10. **The generated-table path macro has three declarations and one dead one.**
    `\resultstables` in `msc_thesis.tex` points at `results/tables`, which does not
    exist; the generated tables live under `media/results/tables`. This pass moved
    the results chapter onto `\resultsfigs` and `\resultsgen`, the pair that
    resolves to the real directory and that the methodology chapter already uses,
    and converted the chapter's last reference to the dead macro. No new
    declaration was added. The remaining cleanup is one line in `msc_thesis.tex`
    and two in `sections/methodology/data.tex`: declare the pair once in the
    preamble and delete `\resultstables`. That is outside this unit, so it was
    left alone.
11. **The corpus-statistics table is entirely fabricated.** It is generated with
    every number invented and belongs to the methodology sections that are marked
    blocked on exactly those numbers. Out of this unit, named here because it is
    on the same outstanding list as the results placeholders.

## Float disposition

Every float in the holding pen now carries its disposition as a one-line comment
in the source. Summarised:

**File into the RQ1 sections.** Parity plot and calibration curve to the accuracy
subsection. Residuals by circuit scale, error by optimization tier, and
performance by optimizability band to the error analysis. Baseline group
comparison, variance decomposition, and the baseline table to the naive-baseline
comparison. Training budget consumed to the cost subsection.

**File into the RQ1a protocol subsection.** Per-design error on unseen designs,
split protocol sensitivity, the protocol table, and the per-design table.

**File into the RQ2 sections.** Node and edge retention, retention distributions,
offline cost, and the amortisation threshold to the offline profile. Peak training
memory and cost against graph size to the memory subsection. Training throughput
and paired per-graph savings to the throughput subsection. The efficiency table to
the ranking subsection.

**File into the RQ3 sections.** Matched-state accuracy, error against explained
variance, accuracy by optimizability band, and the retention table to the
degradation subsection. Matched-state ranking under label strata to the ranking
subsection. Accuracy against achieved compression and accuracy retained to the
attribution subsection. Pareto front figure and table to the Pareto subsection.

**File into the RQ4 sections.** Pairing quality, partitioner trade-off, and the
pairings table to the pairings subsection. Paired gaps, the unpaired view, and the
seed-variance placeholder to the gap subsection.

**File into the RQ5 sections.** Matched-state against full-graph evaluation, the
positive control, and the cross-state table to the accuracy subsection.
Cross-state ranking under label strata to the zero-shot ranking subsection. Cost
of the structural shift and the transfer quadrant to the drop-off subsection.
Inference cost and processor-against-accelerator inference to the cost subsection.

**File into the summary section.** The consolidated table.

**Move to the appendix.** Training trajectories, the validation-to-test gap,
matched-state accuracy by tier and by source script, and the stratified,
subgroup, target-bin, paired-savings and partition-balance tables to the extended
results appendix. The hyperparameter sweep, the hyperparameter sensitivity plot
and the hyperparameter table to the experimental configuration appendix.

**Delete unless the runs land.** The summarization landscape goes with the family
if it is not run. The two receptive-field floats go unless the metric is
implemented. The residual-redundancy probe moves to the dataset statistics
section if the probe runs and is deleted otherwise.

## Placeholder and fabricated inventory

Caption-tagged floats, in the source, with what each is waiting on.

| Float | Tag | Waiting on |
| --- | --- | --- |
| Baseline group comparison | PARTLY FABRICATED | standard encoders under identical training; DeepGate4 and HOGA |
| Split protocol sensitivity | MOSTLY FABRICATED | the random-split and recipe-split training runs |
| Node and edge retention | PARTLY FABRICATED | summarization offline statistics |
| Offline cost per method | PARTLY FABRICATED | summarization offline statistics |
| Matched-state accuracy | PARTLY FABRICATED | summarization training runs |
| Gaps against run-to-run noise | ERROR BARS FABRICATED | repeated seeds on the RQ4 pairs |
| Processor against accelerator inference | MOSTLY FABRICATED | a processor-side inference pass |
| RQ5 positive control | ENTIRELY FABRICATED | the exact colour-refinement track |
| Summarization landscape | ENTIRELY FABRICATED | the whole summarization family |
| Residual redundancy after strashing | ENTIRELY FABRICATED | the refinement-depth probe |
| Effective receptive field | ENTIRELY FABRICATED | the metric, which is not implemented |
| Receptive field against accuracy | ENTIRELY FABRICATED | the same metric |

Generated tables carrying invented rows, each tagged row by row in the table
itself: the RQ1 baselines table, the RQ1a protocol table, the RQ2 efficiency
table, the RQ3 retention table, and the consolidated summary table. The
corpus-statistics table is fabricated in full and belongs to the methodology
chapter.

Separation from real results is currently good. Fabricated floats carry a red
frame, cross-hatching, a watermark and a row-level tag, the captions say so in
bold, and the gallery header lists the outstanding runs. Nothing measured is
presented as fabricated or the reverse. The tagging travels with the float,
since it lives in the caption and in the generated table body rather than in the
gallery scaffolding, so filing a float into its section does not strip it. The
disposition comments deliberately carry no fabrication marker of their own, to
avoid a second place that has to be kept in step with the captions.

## Note on the disposition comments

The 62 disposition comments are outline scaffolding for a holding pen, and they
are written as directives, which the comment rules otherwise keep out of the
submitted source. They come out as each float is filed, and the gallery section
comes out with the last of them. Nothing else in the chapter depends on them.

## Five lines for the supervisor discussion

1. The chapter now has a complete evidentiary plan: each of RQ1, RQ1a, RQ2, RQ3,
   RQ4 and RQ5 has a results home with its exhibits named, and every generated
   float has a filing decision.
2. The single largest hole is the summarization family, which is entirely unrun
   and which four separate claims depend on, including the provably lossless
   coarsening contribution and the RQ5 positive control.
3. RQ4 cannot currently be answered honestly: every configuration is trained once,
   the expected effect is small, and one of its four pairings is not matched at
   equal compression.
4. RQ1's answer is weaker than the headline suggests. A quarter of the label
   variance is carried by source-script grouping alone, and the model predicts
   nearly a constant on the most optimizable circuits, so the chapter needs to
   state plainly which number it defends.
5. Three placeholder groups are cheap to close and unlock disproportionate value:
   two protocol training runs for RQ1a, one processor-side inference pass for RQ5,
   and seeds on the RQ4 pairs only.
