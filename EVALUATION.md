# Running the Evaluation Pipeline

The eval cache is a **separate workspace** (`$EVAL_ROOT`, default
`/scratch-shared/$USER/aig_eval_run`) from the training cache, so test-split
graphs are never written into the train cache. Trained checkpoints are still
read from the train workspace (`aig_train_run/…/checkpoints`) — only the cache
moves. `RUN_ROOT=$EVAL_ROOT` points the mask-precompute at that same workspace
(masks are written in-place there). Both precompute scripts now *default* to
the eval root, so the explicit `RUN_ROOT=` below is redundant — it is kept so
each command states which workspace it writes to. Rebuilding masks for
**training** is the case that now needs `RUN_ROOT=/scratch-shared/$USER/aig_train_run`
passed explicitly.

Masks are never built on demand: a missing mask raises at eval time
(`Precomputed sparsification/partition mask ... not found`) and fails that array
task, rather than silently recomputing them inside the eval job.

## 1. Warm the test-split cache (once)
```bash
export EVAL_ROOT="/scratch-shared/$USER/aig_eval_run"   # separate from train cache
W=$(sbatch --parsable src/shell/warmup_test_cache.sh)

# SPARSIFICATION_ALGO defaults to "all" — data.sparsification computes all 4
# methods (and_gate_only, random_edge_dropout, spanning_forest, pagerank) in
# one pass, matching precompute_partition_masks.sh's own "all" default. Env-
# var overrides on this cluster MUST use `--export=ALL,VAR=value` — a bare
# `VAR=value sbatch ...` prefix does not reliably propagate on Snellius and
# will silently fall back to the script's own default instead of erroring.
S=$(sbatch --parsable --export=ALL,RUN_ROOT=$EVAL_ROOT --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)

M=$(sbatch --parsable --export=ALL,RUN_ROOT=$EVAL_ROOT --dependency=afterok:$W src/shell/precompute_partition_masks.sh)  # "all" partition methods in one pass
```

## 2. Run inference eval (array jobs, 9 configs each)
```bash
DEPS="afterok:$S:$M"
sbatch --dependency=$DEPS src/shell/test.sh       # GPU: accuracy + inference hardware
sbatch --dependency=$DEPS src/shell/test_cpu.sh   # CPU: inference hardware only
```

Run the inference eval **first** and let it finish before starting step 3. It
is the step the accuracy results (RQ1–RQ3) depend on, it is the one gated on
the cache/mask chain above, and `test.sh` is the only GPU-partition job here —
so getting it through the queue first keeps the training benchmark from sitting
in front of it competing for GPU budget.

### Evaluating a non-default split strategy

The steps above evaluate `config.SPLIT_BY` (`design`). To evaluate a checkpoint
trained with `train.py --split_by recipe|random` (e.g. `sbatch -a 0-2
src/shell/train_no_sparsification.sh`, whose tasks 0/1/2 are design/recipe/random),
pass the same value through the **whole** chain — the warmup included, since a
different strategy puts different graphs in the test split:

```bash
sbatch --parsable --export=ALL,SPLIT_BY=recipe src/shell/warmup_test_cache.sh
sbatch --dependency=$DEPS --export=ALL,SPLIT_BY=recipe src/shell/test.sh
```

`test.py` then evaluates that split and reads the `<run_label>_recipe`
checkpoint directory `train.py` wrote to; its result/prediction filenames and
WandB run names carry the same suffix, so they never collide with the design
run. The warmup sentinel is keyed on `SPLIT_BY` too, so a prior design warmup
does not make the recipe warmup skip itself.

### Batching at eval time

Eval packs batches to a total-node budget — the same scheme training uses —
rather than a fixed 32 graphs per batch. AIGs range up to ~366k nodes, so a
fixed graph count both wastes the GPU on runs of small graphs and risks OOM on
a run of large ones.

The settings live in `config.py`, **not** in the SLURM scripts, so all 9 array
tasks and both devices cannot drift onto different batchings:

| `config.py` | default |
|---|---|
| `EVAL_DYNAMIC_BATCHING` | `True` |
| `EVAL_MAX_TOTAL_NODES_PER_BATCH` | `8_000_000` |
| `EVAL_PREFETCH_FACTOR` | `2` (below training's 4 — in-flight host memory is `num_workers × prefetch_factor` batches, and eval batches are much larger) |

What this does and does not change:

| | Effect of node-budget batching |
|---|---|
| Accuracy (Smooth L1 / RMSE / R² / Spearman) | **None.** Metrics are computed over the whole concatenated prediction tensor, the model normalises per-node (`NORM_TYPE="layer"`) and pools per-graph, so batch composition cannot move them. Covered by `TestRunEvalPassBatchingInvariance`. |
| `throughput_graphs_per_s` | Improves, and becomes where the reduction benefit shows up (fewer nodes ⇒ more graphs per batch). |
| `peak_vram_mb` | Becomes ~constant across configs **by construction** — the budget holds nodes/batch fixed. Don't read a reduction's memory benefit off this column while dynamic batching is on. |

The 8M budget sits above training's 3M because forward-only eval holds no
gradients or optimizer state. It was set from a measured 5M run on an H100
80GB, which peaked at ~40% GPU memory (~34GB, i.e. ~6.8GB per 1M nodes); 8M
projects to ~54GB (~68%), leaving headroom for allocator fragmentation and for
a peak falling between NVML samples. Watch `peak_vram_mb` and
`peak_process_rss_mb` on the first run at the new value:

- if `peak_vram_mb` lands well under ~55GB there is room for ~10M;
- `peak_process_rss_mb` is the one that can bite first — in-flight *host*
  memory is `num_workers × prefetch_factor` batches (12 × 2 = 24), each now
  1.6× larger, and neither eval SLURM script requests an explicit `--mem`.
  Drop `EVAL_PREFETCH_FACTOR` to 1 before dropping the node budget.

Don't expect throughput to scale with the budget. That same 5M run showed SM
Active ~100% at 82% occupancy with the FP32 pipeline at ~5% and DRAM at ~37% —
the kernels are latency-bound on message-passing gather/scatter, not compute-
or bandwidth-bound, so a larger budget only amortizes per-batch collate/H2D/
launch overhead.

The eval batch plan is rebuilt in memory each run (`AIGDataModule._ensure_test_plan`
calls `build_batch_plan` directly and does not touch the on-disk plan cache), so
a budget change takes effect immediately with no stale-plan risk. The *training*
plan is disk-cached, but `batch_plan_cache_path` hashes `max_total_nodes` into
the filename, so that one is safe too.

The effective setting lands in each row's single `batching` column
(`dynamic_nodes=8000000` or `fixed_graphs=32` — only one of the two knobs is
ever live), so a mismatch across configs is visible in the CSVs rather than
silent. **Changing it means re-running every config**, not just the new one,
or the hardware columns stop being comparable.

`throughput_graphs_per_s` excludes the first batch (CUDA context init, cuDNN
autotune and worker spawn all land there). `num_timed_graphs` records how many
graphs the timing covers, and `total_time_s` covers that same region — so
divide by `num_timed_graphs`, not `num_graphs`. Both go NaN when a pass fits in
a single batch, since there is then no steady-state region at all.

### Validation-set sanity pass

Every `test.sh` array task also runs one extra pass on the **validation**
split, in the same reduction form training's own val loop used for that
config (`AIGDataModule` applies `sparsification`/`partition` identically to
`train_ds` and `val_ds` — see `datamodule._make_dataset`) — a check that this
eval pipeline reproduces training's own reported val metrics, not a thesis
result (excluded from `results_to_latex.py`'s tables via the `split=="test"`
filter). It reads from the **train** workspace, not `$EVAL_ROOT`: validation
graphs and their reduction masks were built there by `warmup_train_cache.sh` /
`precompute_*_masks.sh RUN_ROOT=aig_train_run` — `warmup_test_cache.sh` only
ever warms the test split under `$EVAL_ROOT`. `test.sh` wires this via
`--val_cache_dir`/`--val_tier0_cache_dir`/`--val_tier1_cache_dir`.

`test_cpu.sh` passes `--skip_val true` — it only measures inference hardware,
and the sanity pass is already covered by the GPU run. Disable it for a single
`test.sh` invocation with `EXTRA_ARGS="--skip_val true"`.

### WandB

Each array task opens one run named `test_<algorithm>[_<type>_<method>]`,
mirroring train.py's `train_<…>` scheme, into the same project/entity (now
`config.WANDB_PROJECT` / `config.WANDB_ENTITY`, shared by both scripts).
Metrics land in the run summary prefixed by split then pass —
`test/full_graph/rmse`, `val/matched_reduction/throughput_graphs_per_s`, …
(the split prefix matters: test and val can share an `eval_mode` string, e.g.
both `full_graph` for a baseline config, and would otherwise overwrite each
other's summary entries). The CSV write happens *first*, so WandB is a mirror
and never the source of truth. Each pass's per-graph predictions CSV (when
`--dump_predictions true`) is additionally uploaded as a WandB Artifact named
after the CSV's filename, so per-graph results are recoverable from the run
without cluster filesystem access — a Table isn't used since a full test
split is too many rows for its render/size limits. Disable WandB entirely with
`WANDB=false sbatch …`.


## 3. Training-hardware benchmark (after step 2)
```bash
sbatch src/shell/benchmark.sh                     # no cache dependency
```

Ordering is a scheduling choice, not a correctness one: `benchmark.sh` has no
cache or mask dependency and would run correctly at any point.

The benchmark measures **one graph per batch** (not real training's node-budget
dynamic batching, which holds per-batch VRAM ~constant across methods and hides
the memory benefit) so per-graph VRAM/latency is comparable full-vs-reduced.

## 4. Turn results into thesis tables/figures
```bash
PYTHONPATH=src python -m results_to_latex   # results/tables/*.tex + pareto_front.csv
PYTHONPATH=src python -m plot_results       # results/figures/*.png
```

Re-running a single config cleanly overwrites its own result files (below), so
you can re-run one array index without duplicating rows.

## Outputs
Each array task writes its **own** file (not a shared appended CSV) to avoid
concurrent-write races; `results_to_latex.py` / `plot_results.py` glob these
directories.

| Path | Contents |
|---|---|
| `results/inference_results/*.csv` | accuracy + inference hardware, one file per (config, eval_mode, split, device) — non-test splits get a `_val` filename suffix and `split` column, and are excluded from the RQ tables |
| `results/training_benchmark/*.csv` | training hardware aggregate, one file per config (one graph per batch) |
| `results/benchmark_per_graph/*.csv` | per-graph training step time + VRAM, one file per config (enables paired full-vs-reduced analysis) |
| `results/predictions/*.csv` | per-graph predictions (GPU pass only, `_val` suffix for the validation sanity pass); also uploaded as a WandB Artifact per file |
| `results/tables/*.tex` | RQ1–RQ4 tables — `\input{}` into `sections/4-results.tex` |
| `results/figures/*.png` | parity plots, RQ2/RQ3/RQ4 charts, Pareto front, VRAM-vs-size + VRAM-saving-vs-size scatters |
