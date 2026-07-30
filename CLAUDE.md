# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral guidelines

Bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
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

Before adding a new constant, helper, or code path, check whether an equivalent already exists (this repo has real precedent for drift — see `config.py` vs `constants.py` in [src/CLAUDE.md](src/CLAUDE.md)). Reuse or consolidate rather than adding a second source of truth.

### 4. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
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

Any time code has been written or edited in a session, before running `git commit`/`git push`, spawn a fresh sub-agent with no prior context to act as an adversarial reviewer of the changes made (diff only, no memory of why the change was made). Address what it flags — fix real issues, or note why a flagged item is a non-issue — before committing.

---

## Project overview

Thesis project building a dataset + GNN regression pipeline for And-Inverter Graphs (AIGs) — hardware circuit netlists. The pipeline has two distinct halves that don't share code:

1. **Data generation** (`data/creation/`): shell + Python scripts that drive the `abc` logic-synthesis tool to optimize benchmark circuits with four algorithms (`Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`) and export metadata CSVs. See [AIG_DATASET_README.md](AIG_DATASET_README.md) and [abc.rc](abc.rc) for per-algorithm command templates/aliases.
2. **ML pipeline** (`src/`): loads the generated AIGs as PyTorch Geometric graphs, optionally sparsifies them, and trains a GNN (Lightning) to regress "node optimizability."

**Everything runs on an HPC SLURM cluster (Snellius: `module load 2025`, partitions `gpu_h100`/`genoa`, `/scratch-shared/$USER` paths) — nothing is expected to run on a laptop.** All training data and caches live on cluster scratch storage, not in this repo. `src/shell/*.sh` are SLURM job scripts (`sbatch ...`); treat any "run this" request as "prepare/edit the job script," not "execute it here," unless the user is explicitly on a login/interactive node.

Training currently only targets the `Orchestrate` synthesis algorithm (`config.VALID_ALGORITHMS = {"Orchestrate"}`) — `Deepsyn`/`Syn4`/`C2RS` graphs still exist on disk from the data-generation pipeline but aren't used for training right now.

Commands and architecture notes for the `src/` ML pipeline live in [src/CLAUDE.md](src/CLAUDE.md) — loaded automatically when working under `src/`.
