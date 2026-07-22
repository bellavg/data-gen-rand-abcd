# Running the Evaluation Pipeline

## 1. Warm the test-split cache (once)
```bash
W=$(sbatch --parsable src/shell/warmup_test_cache.sh)

# One invocation per sparsification method — SPARSIFICATION_ALGO defaults to
# and_gate_only, so all 4 must be run explicitly or the other 3 configs will
# crash at eval time with "mask not found".
S1=$(SPARSIFICATION_ALGO=and_gate_only        sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S2=$(SPARSIFICATION_ALGO=random_edge_dropout  sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S3=$(SPARSIFICATION_ALGO=spanning_forest      sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
S4=$(SPARSIFICATION_ALGO=pagerank             sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)

M=$(sbatch --parsable --dependency=afterok:$W src/shell/precompute_partition_masks.sh)  # "all" partition methods in one pass
```

## 2. Run eval (array jobs, 9 configs each)
```bash
DEPS="afterok:$S1:$S2:$S3:$S4:$M"
sbatch --dependency=$DEPS src/shell/test.sh       # GPU: accuracy + inference hardware
sbatch --dependency=$DEPS src/shell/test_cpu.sh   # CPU: inference hardware only
sbatch src/shell/benchmark.sh                     # controlled training-hardware benchmark (no cache dependency)
```

## 3. Turn results into thesis tables/figures
```bash
PYTHONPATH=src python -m results_to_latex   # results/tables/*.tex + pareto_front.csv
PYTHONPATH=src python -m plot_results       # results/figures/*.png
```

## Outputs
| Path | Contents |
|---|---|
| `results/inference_results.csv` | accuracy + inference hardware, one row per (config, eval_mode) |
| `results/training_benchmark.csv` | training hardware, one row per config |
| `results/predictions/*.csv` | per-graph predictions (GPU pass only) |
| `results/tables/*.tex` | RQ1–RQ4 tables — `\input{}` into `sections/4-results.tex` |
| `results/figures/*.png` | parity plots, RQ2/RQ3/RQ4 charts, Pareto front |
