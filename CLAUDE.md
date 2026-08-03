# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral guidelines

Bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them rather than picking one silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First (YAGNI)

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Don't Repeat Yourself (DRY)

Before adding a new constant, helper, or code path, check whether an equivalent already exists (this repo has real precedent for drift: see `config.py` vs `constants.py` in [src/CLAUDE.md](src/CLAUDE.md)). Reuse or consolidate rather than adding a second source of truth.

### 4. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it but do not delete it.
- Remove imports/variables/functions that YOUR changes made unused. Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 5. Goal-Driven Execution

Define success criteria. Loop until verified.

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

### 6. Adversarial Review Before Commit

Any time code has been written or edited in a session, before running `git commit`/`git push`, spawn a fresh sub-agent with no prior context to act as an adversarial reviewer of the changes made (diff only, no memory of why the change was made). Before committing, address what it flags: fix real issues, or note why a flagged item is a non-issue.

---

## Project overview

Thesis project building a dataset + GNN regression pipeline for And-Inverter Graphs (AIGs), i.e. hardware circuit netlists. The pipeline has two distinct halves that don't share code:

1. **Data generation** (`data/creation/`): shell + Python scripts that drive the `abc` logic-synthesis tool to optimize benchmark circuits with four algorithms (`Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`) and export metadata CSVs. See [AIG_DATASET_README.md](AIG_DATASET_README.md) and [abc.rc](abc.rc) for per-algorithm command templates/aliases.
2. **ML pipeline** (`src/`): loads the generated AIGs as PyTorch Geometric graphs, optionally sparsifies them, and trains a GNN (Lightning) to regress "node optimizability."

**Everything runs on an HPC SLURM cluster (Snellius: `module load 2025`, partitions `gpu_h100`/`genoa`, `/scratch-shared/$USER` paths). Nothing is expected to run on a laptop.** All training data and caches live on cluster scratch storage, not in this repo. `src/shell/*.sh` are SLURM job scripts (`sbatch ...`); treat any "run this" request as "prepare/edit the job script," not "execute it here," unless the user is explicitly on a login/interactive node.

Training currently only targets the `Orchestrate` synthesis algorithm (`config.VALID_ALGORITHMS = {"Orchestrate"}`). The `Deepsyn`/`Syn4`/`C2RS` graphs still exist on disk from the data-generation pipeline but aren't used for training right now.

Commands and architecture notes for the `src/` ML pipeline live in [src/CLAUDE.md](src/CLAUDE.md), loaded automatically when working under `src/`.

---

## Thesis writing (`IV_Gardner___Master_AI_Thesis_Outline/`)

Register, voice, citation, and formatting rules for `.tex` prose live in [IV_Gardner___Master_AI_Thesis_Outline/CLAUDE.md](IV_Gardner___Master_AI_Thesis_Outline/CLAUDE.md), loaded automatically when working under that directory.

### Overleaf mirror, and what stays private

This directory is published to a separate private repo, `git@github.com:bellavg/msc-thesis-latex.git`, which Overleaf imports so the supervisor can read the thesis. Nothing about this repo's structure changes: the files here stay ordinary tracked files, and the mirror only moves when the publish script is run.

```bash
bash IV_Gardner___Master_AI_Thesis_Outline/sync-overleaf.sh
```

The publish takes the committed state of whatever branch is checked out (normally `thesis-outline`), so uncommitted edits stay local. It overwrites the mirror, so treat Overleaf as read-only and do the writing here.

### Target workflow: `thesis-outline` is rough, `main` is clean (not active yet)

The intent, not yet acted on: `thesis-outline` stays the working branch, full of `% TODO` lines and the `% Original, condensed above:` preserved-prose blocks described below. `main` is meant to hold only finished, comment-free text, section by section, as each one is declared done. `sync-overleaf.sh` should eventually read from `main` instead of whatever branch is checked out, so Overleaf and the supervisor only ever see cleaned sections.

This isn't running yet: as of 2026-08-03, `main` is 55 commits behind `thesis-outline` on thesis content and holds nothing current, and no section has been declared final. Don't merge or promote anything to `main` on your own initiative. "Final" is her call per section, made explicitly, not inferred from silence or from how polished a section looks.

Once she names a section as final:
1. Sweep that section's `% TODO` lines and `% Original, condensed above:` blocks, leaving only live text.
2. Commit the cleaned section to `main` (cherry-pick or copy the file).
3. Leave `thesis-outline`'s full rough version untouched.
4. Once `main` actually carries current content, point `sync-overleaf.sh` at it instead of the checked-out branch.

### Planning and notes files stay between the author and Claude

Working documents written for Claude or with Claude are private and are never shown to the supervisor or to any other reader: `CLAUDE.md` files, `HANDOFF.md`, writing plans, open-question lists, review notes, per-section scratch files like `sections/1-introduction/terms-to-introduce.md`, and anything similar written later. Treat a new one as private without being told.

The publish script enforces this with an allowlist, not a blocklist. Its `PUBLISH` array names the only paths that ever leave the repo (`msc_thesis.tex`, `mscaithesis.cls`, `README.md`, `sections/**/*.tex`, `bibliographies/**/*.bib`, `media/`), so a notes file added anywhere under this directory is private automatically, with nothing to remember and nothing to add to a list. Keep that default when changing the script: add a file type or a directory, never carve out an exception for an individual notes file, and never switch it to "publish everything except...".

Two things are deliberately excluded and should stay excluded. `thesis-overview.tex` is a planning document carrying supervisor comment threads and a superseded research-question list, and it is a second `\documentclass` root that would confuse Overleaf's main-document detection. `msc_thesis.pdf` and the rest of the build output are regenerated by Overleaf.

**Never use an em dash (—) or an en dash used as punctuation (–). Zero, anywhere, no exceptions.** The author dislikes them and reads them as a tell for generated text. Rewrite with a comma, a colon, parentheses, or a new sentence. This applies to `.tex` prose, to this file, to any document written for her, and to chat replies. (En dashes in numeric ranges, "2020–2024", and LaTeX's `--`/`---` in verbatim or citation macros are unaffected.)
