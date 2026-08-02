# Thesis outline batch: handoff

Working note, not part of the build. Written 2026-08-02.

## Where things stand

Branch `thesis-outline`, builds clean: 0 errors, 0 undefined references, 0 undefined
citations, 0 broken figure targets, 0 em or en dashes. Verify with:

```bash
cd IV_Gardner___Master_AI_Thesis_Outline && bash check-build.sh
```

Chapters were reorganised into numbered per-chapter folders. Subsection `\input` calls carry
an explicit `.tex`, because LaTeX reads the dot in `1.1-name` as a file extension.

```
sections/0-frontmatter/{0.1-abstract, 0.2-list-symbols}
sections/1-introduction/{1-introduction, 1.1-motivation, 1.2-research-questions,
                         1.3-contributions, 1.4-outline, terms-to-introduce.md}
sections/2-related-work/{2-related-work, 2.1-prelim-synthesis ... 2.12-relwork-gap}
sections/3-methodology/{3-methodology, 3.1-architecture ... 3.9-experiment-reproducibility}
sections/4-results/{4-results, 4.1-rq1 ... 4.6-summary}
sections/5-discussion/, sections/6-conclusion/, sections/7-appendix/
```

Live prose per chapter: frontmatter 225, introduction 1,901, related work 5,844,
methodology 9,360, results 4,091, discussion 2,595, conclusion 666, appendix 533.
Total 25,215, down from 33,154. **The goal at this stage is showing the supervisor what
content will be in the thesis, not hitting the page limit.** She confirmed this directly:
the condensed text is the deliverable, he has little time, and he is not reviewing prose
style or flow. Length is not the constraint; content completeness is.

A further 13,000-odd words sit in `%` comments across chapters 2 and 3. **That is
deliberate and must stay.** Cut prose is preserved in place so she can use the full text in
the real thesis. Never sweep those blocks.

## The two tasks running at the last handoff: resolved

Both died before committing. Their work was recovered from their worktrees and is in
`bbd4109`. One had finished `3.6-data` only; the other `3.1-architecture` and
`3.2-baselines` only. All three files were byte-identical between the agents' base and
HEAD, so recovery was a file copy, not a merge.

**The recovered work was not safe.** An adversarial review of it found eleven defects,
including three numerals that survived condensation while the population they describe
changed, two of which were false as written:

- `4.8%` was the size of the step-21 subset (200 of 4,201 graphs per design), not the share
  of the corpus carrying the wrong denominator, which is 43 of 4,201, near `1%`.
- `37-39%` primary outputs is of the **gate count**, not of nodes. Against all nodes it is
  33-34%.
- median depth 31 levels holds over the 47 designs outside the **EPFL suite**; the condensed
  wording read as the 37 non-classical designs, whose median is 29.

Fixed in `bbd4109`, each verified against `media/tables/corpus_designs.tex`. The remaining
eight defects are fixed in `7aecd3d`. **Lesson for any future condensation pass on this
document: check every sentence containing a number against the population it described
before, not just that the numeral survived.**

The other worktree-agent branches are stale. They predate the numbered-folder
reorganisation (139 files, ~5,349 deletions each) and merging any of them would delete the
current layout.

## What is left

**Chapter 2 length: closed.** She declined all three levers. Its condensation is also
verified faithful: a mechanical check found zero quantities that appear only in comments, and
a claim-by-claim audit of `2.1`, `2.2` and `2.3` found nothing missing. Do not reopen it.

**Results, introduction, discussion and conclusion condensation: closed.** She reframed the
goal to content completeness rather than length, so these stay at full density.

**Scope and Delimitations: done**, 47 lines to 37, seven items to four, all seven exclusions
still stated. It did not reach the 15 to 20 lines `scope-section-exemplars.md` recommends and
**that target is not reachable** with the content intact: 14 of the 37 lines are structural
and the 9 forward references are unbreakable markup, putting the floor near 35. Getting to 20
requires dropping the forward references or an exclusion.

**"State": done, and it was two terms, not one.** *Structural state* (the form a graph
reaches the encoder in, reduced or unreduced) is defined at RQ5. *Optimization state*
(whether a target synthesis algorithm has already been applied) is glossed in the
contributions list. Both were previously undefined and are easy to confuse.

**Two-panel figure titles: done**, and the cause was not the `WIDE` height. `apply_style()`
set no layout engine at all, so `savefig`'s `bbox_inches="tight"` cropped whitespace without
repositioning titles, and every `suptitle` compensated with a hand-tuned `y=` offset.
`figure.constrained_layout.use` in the shared rcParams fixed all 20 two-panel figures at
once; four now-dead `subplots_adjust` calls were removed and two over-long title pairs
wrapped.

## Still open

1. **`3.6-data.tex` needs a content read, not another cut.** It has had a rework and a
   condensation pass, is the longest section in the chapter, and carries the largest block of
   commented prose (about 3,500 words). Seven of the eleven adversarial-review defects were
   in this file. A dozen lower-severity drops in it were deliberately deferred and still need
   her judgement, the sharpest being a `33.8%` to `87.8%` range whose unit ("of a design's
   edges") is no longer stated.
2. **Final build plus `bash make-latexdiff.sh`** for a marked-up PDF against `main`.
3. **`rq3_pareto`**: the "edge-drop" point label overlaps the "full" star label inside the
   left panel. Pre-existing, unrelated to the title bug, not fixed.
4. **`thesis-overview.tex`** is a stale standalone document, not wired into the build. Its
   research questions were reconciled with the thesis, but its prose still predates several
   of the register rules and it embeds the supervisor's own earlier feedback in comments,
   initialled "M:", which is worth reading: he asked for higher-level motivation and for
   results to be teased there.

## Deliberately not done

- **The `%` comment sweep** that `CLAUDE.md` asks for before sending work out. Those blocks
  hold the full prose she asked to keep commented out during condensation. Sweeping now would
  delete exactly what she asked to preserve. That belongs at submission.
- **Committing the root `CLAUDE.md` change.** 124 lines were moved out of it into
  `IV_Gardner___Master_AI_Thesis_Outline/CLAUDE.md`; the deletion is still uncommitted. It is
  her editorial change to her own instructions file.

## Markers still in the source, all justified

Eight `\todo` macros and about 45 one-line `% TODO` comments remain. Every one is blocked on
a run that does not exist, and each states what would resolve it. The five in
`6-conclusion.tex` and the one in `0.1-abstract.tex` are waiting on results numbers. The
largest cluster, fifteen in `3.5-reduction-summarization.tex`, reflects that no summarization
configuration has been trained.

Two are real open problems rather than waiting on a run:

- **The synthesis tool version was never recorded** and cannot be recovered. No submodule, no
  lockfile, no captured version string, and the module loads cover only `foss`, `Python` and
  `SciPy`. The appendix and the bibliography both state this as a reproducibility bound.
- **Step-21 graphs are still scored with the wrong denominator.** They are choice networks,
  not plain AIGs, so the tool-reported node count covers only the chosen representative, in
  43 of 200 recipes, about 4.8 percent of the corpus. They remain in every split. The planned
  correction is to drop them from reported inference scores rather than relabel and retrain.
  Not implemented in the pipeline.

## Gotchas that cost time in this batch

- **New worktrees are often created on the wrong base.** Roughly a third landed on the
  ML-pipeline lineage instead of `thesis-outline`. Always have an agent run
  `git log --oneline -3` first and `git reset --hard thesis-outline` if it sees commits about
  "eval split_by" or "SLURM array".
- **Do not commit `msc_thesis.pdf`.** The build rewrites it and parallel branches conflict on
  the binary every time. `git checkout -- IV_Gardner___Master_AI_Thesis_Outline/msc_thesis.pdf`
  before committing.
- **The `code-review` skill is not invocable from a subagent.** Ask for a by-hand adversarial
  re-read of the diff instead.
- **Use the project venv for the analysis pipeline**: `.venv/bin/python -m src.analysis.make_all`.
  Plain `python3` has no pandas.
- **`check-build.sh` reports a chktex count against a stale baseline of 154.** It does not gate
  on it. The real gates are errors, undefined references and citations, broken targets, dashes.
