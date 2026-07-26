# Running the Evaluation Pipeline

The eval cache is a **separate workspace** (`$EVAL_ROOT`, default
`/scratch-shared/$USER/aig_eval_run`) from the training cache, so test-split
graphs are never written into the train cache. Trained checkpoints are still
read from the train workspace (`aig_train_run/…/checkpoints`) — only the cache
moves. `RUN_ROOT=$EVAL_ROOT` points the mask-precompute at that same workspace
(masks are written in-place there).

## 1. Warm the test-split cache (once)
```bash
export EVAL_ROOT="/scratch-shared/$USER/aig_eval_run"   # separate from train cache
W=$(sbatch --parsable src/shell/warmup_test_cache.sh)

# One invocation per sparsification method — SPARSIFICATION_ALGO defaults to
# and_gate_only, so all 4 must be run explicitly or the other 3 configs will
# crash at eval time with "mask not found".
S1=$(RUN_ROOT=$EVAL_ROOT SPARSIFICATION_ALGO=and_gate_only        sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S2=$(RUN_ROOT=$EVAL_ROOT SPARSIFICATION_ALGO=random_edge_dropout  sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S3=$(RUN_ROOT=$EVAL_ROOT SPARSIFICATION_ALGO=spanning_forest      sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S4=$(RUN_ROOT=$EVAL_ROOT SPARSIFICATION_ALGO=pagerank             sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)

M=$(RUN_ROOT=$EVAL_ROOT sbatch --parsable --dependency=afterok:$W src/shell/precompute_partition_masks.sh)  # "all" partition methods in one pass
```

## 2. Run eval (array jobs, 9 configs each)
```bash
DEPS="afterok:$S1:$S2:$S3:$S4:$M"
sbatch --dependency=$DEPS src/shell/test.sh       # GPU: accuracy + inference hardware
sbatch --dependency=$DEPS src/shell/test_cpu.sh   # CPU: inference hardware only
sbatch src/shell/benchmark.sh                     # controlled training-hardware benchmark (no cache dependency)
```

The benchmark measures **one graph per batch** (not real training's node-budget
dynamic batching, which holds per-batch VRAM ~constant across methods and hides
the memory benefit) so per-graph VRAM/latency is comparable full-vs-reduced.

## 3. Turn results into thesis tables/figures
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
| `results/inference_results/*.csv` | accuracy + inference hardware, one file per (config, eval_mode, device) |
| `results/training_benchmark/*.csv` | training hardware aggregate, one file per config (one graph per batch) |
| `results/benchmark_per_graph/*.csv` | per-graph training step time + VRAM, one file per config (enables paired full-vs-reduced analysis) |
| `results/predictions/*.csv` | per-graph predictions (GPU pass only) |
| `results/tables/*.tex` | RQ1–RQ4 tables — `\input{}` into `sections/4-results.tex` |
| `results/figures/*.png` | parity plots, RQ2/RQ3/RQ4 charts, Pareto front, VRAM-vs-size + VRAM-saving-vs-size scatters |
