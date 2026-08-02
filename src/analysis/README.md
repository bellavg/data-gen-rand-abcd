# `src/analysis` — Results-chapter figures and tables

Everything the thesis Results chapter renders is built from here. Nothing in
this package trains, evaluates, or touches the cluster; it only reads the
exported result CSVs.

## Build everything

```bash
PYTHONPATH=src python -m analysis.make_all
```

Writes `IV_Gardner___Master_AI_Thesis_Outline/media/results/figures/*.pdf` (44
figures) and `.../tables/*.tex` (14 tables). Override with `--results-dir` and
`--media-dir`.

## Getting the input data

The result CSVs are gitignored; the tracked copies are the tarballs in
`results/archives/` on `main`. Extract them first:

```bash
for f in results/archives/*.tar.gz; do tar xzf "$f" -C results; done
```

Two inputs are **not** in the archives and have to be fetched separately:

| Input | Where it comes from |
| --- | --- |
| `results/wandb_export/runs.csv`, `train_history.csv` | W&B project `AIG-SUMMARIZE`; `train.py` writes no CSV, so W&B is the sole source of the training curves |
| `results/hp_tuning/*.log`, `*.out` | the Optuna worker logs; the study SQLite databases were lost with the scratch workspaces |

## Layout

| Module | What it holds |
| --- | --- |
| `style.py` | palette, rcParams, the method registry, and the fabricated-data marking |
| `loaders.py` | CSV loading, graph-id parsing, W&B export, Optuna log parsing |
| `fake_data.py` | **every invented number in the figure set**, one block per outstanding run |
| `fig_dataset.py` | corpus and label distributions |
| `fig_rq1.py` … `fig_rq5.py` | one module per research question |
| `fig_placeholders.py` | figures with no data behind them at all |
| `tables.py` | booktabs tables |
| `make_all.py` | entry point |
| `results_to_latex.py` | the original table generator; its loaders and `build_paired_savings` are reused throughout |
| `plot_results.py` | the original figure script; superseded by the `fig_*` modules above, kept because nothing has been checked against it yet |

## Conventions

**Colour encodes the reduction family, never the individual method.** There are
nine measured configurations and six more waiting on runs; a nine-hue
categorical palette is unreadable and colourblind-unsafe, and family is the
organising concept anyway. Individual methods are identified by axis position
and direct labels. Domain-informed methods are hatched rather than recoloured,
because generic-vs-domain is an attribute of a method rather than a second
colour axis.

**Fabricated data is marked five ways**, so that no single loss of context can
turn a placeholder into a result:

1. the bar or point is red and cross-hatched;
2. the row label carries `[FAKE]` (`[TODO/FAKE]` in tables);
3. the figure gets a red frame, a diagonal `TODO / FAKE DATA` watermark
   (omitted when only some rows are invented — the rest are real), and a footer
   naming the run that would replace it;
4. `fake_data.py` holds the numbers, so `grep TODO_ src/analysis/fake_data.py`
   lists everything outstanding;
5. in tables only, the whole row is typeset red and every number in it is
   replaced by the sentinel `999999` (negated where higher is better), so a
   placeholder cannot be read as a result at all. Figures keep the plausible
   values, because a sentinel destroys every axis it lands on.

Delete the block from `fake_data.py` as soon as the corresponding run lands.

## What is still fabricated

| Block | Waiting on |
| --- | --- |
| `SUMMARIZATION` | the entire summarization family — no method has been trained or measured |
| `BASELINE_MODELS` | the DeepGate4 / HOGA ports |
| `SPLIT_PROTOCOL` | RQ1a: `--split_by random` and `--split_by recipe` training runs |
| `RECEPTIVE_FIELD` | the $k$-hop fanin-cone metric, which is specified but not implemented |
| `WL_DEPTH` | the colour-refinement depth probe |
| `CPU_INFERENCE` | `src/shell/test_cpu.sh` — every surviving inference CSV is `device=cuda` |
| `CORPUS_STATS` | a stats pass over the whole graph cache (the predictions cover the eval splits only) |
| `seed_variance()` | 3 seeds on the four RQ4 pairings; every configuration is currently trained once |
