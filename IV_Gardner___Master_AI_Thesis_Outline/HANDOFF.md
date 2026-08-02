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

Live prose per chapter: introduction 1,907, related work 6,604, methodology 14,729,
results 6,652, discussion 2,596, conclusion 666. Total 33,154, roughly 66 two-column pages
against a 35-page limit. **The goal at this stage is showing the supervisor what content
will be in the thesis, not hitting the page limit.** Length matters later.

## Two tasks were still running at handoff

Both are methodology condensation, both in git worktrees, both committed to their own branch
without pushing. Merge each into `thesis-outline` and run `check-build.sh` after each:

```bash
git branch --list 'worktree-agent-*'
git merge --no-edit <branch>
```

- one owns `3.6-data`, `3.7-experiment-setup`, `3.8-experiment-metrics` (8,357 words, target ~3,700)
- one owns `3.1-architecture`, `3.2-baselines`, `3.3`/`3.4`/`3.5-reduction-*` (6,253 words, target ~3,200)

If either died before committing, its work is still in its worktree, uncommitted. Recover it
with `git -C .claude/worktrees/agent-<id> status --porcelain` and copy the changed files
across. This happened repeatedly during the batch and nothing was lost that way.

After merging, verify nothing was dropped:

```bash
# per file, compare against the pre-merge revision
git show HEAD~1:<path> | grep -oE '\\cite[tp]?\{[^}]+\}' | sort -u > /tmp/a
grep -oE '\\cite[tp]?\{[^}]+\}' <path> | sort -u > /tmp/b
diff /tmp/a /tmp/b
```

## What is left

1. **Chapter 2 length, if wanted.** Prose compression is finished. Two independent passes
   stopped at 20 to 28 percent and both gave the same reason: what remains is definitions,
   not padding. Further reduction is a content decision. Three levers, in order:
   - move the Deepsyn, Syn4 and C2RS detail in `2.3-prelim-algorithms.tex` to the appendix
     (532 words on algorithms that are not trained on);
   - merge `2.5-prelim-reduction.tex` and `2.9-relwork-reduction.tex`, which tell one story
     split across preliminaries and related work;
   - drop citations. Her call only.
2. **Results, introduction, discussion and conclusion condensation**, if length matters:
   results 6,652 words and the other three 5,169 between them. Not started.
3. **Scope and Delimitations final trim.** `scope-section-exemplars.md` recommends 15 to 20
   lines and four to five items, down from the current seven, keeping the bold lead-in format
   rather than switching to bullets, and merging two pairs of items that each split one
   exclusion in two. Currently at 47 lines.
4. **Define "state" on first use.** RQ5 was renamed to Cross-State Inference and the term is
   used on roughly thirty lines, but never given a one-line definition.
5. **Two-panel figure titles collide.** Pre-existing layout bug in several `figsize=WIDE`
   figures, confirmed in `dataset_zero_inflation.pdf` and `rq2_throughput.pdf`.
6. **Final build plus `bash make-latexdiff.sh`** for a marked-up PDF against `main`.

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
