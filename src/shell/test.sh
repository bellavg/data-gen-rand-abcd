#!/bin/bash
#SBATCH --job-name=aig_test_gpu_array
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --array=0-8
#SBATCH --output=logs/test_gpu_%A_%a.out

# ---------------------------------------------------------------------------
# GPU inference eval — accuracy (Smooth L1/RMSE/R2/Spearman) + inference
# hardware (throughput, peak VRAM, GPU util, host memory) on the complete
# test split, for each of the 9 trained configs (baseline, 4 sparsification,
# 4 partition methods). Each array task runs test.py once, which itself runs
# the full-graph pass (always) plus the matched-reduction pass (for
# non-baseline configs) on the TEST split, plus one automatic sanity pass on
# the VALIDATION split in this config's own trained-on reduction form (a
# check that this pipeline reproduces training's own reported val metrics,
# not a thesis result — see test.py's module docstring). The val pass reads
# from the TRAIN workspace (VAL_CACHE_DIR below), not $EVAL_ROOT: val's
# graphs and reduction masks were built there by warmup_train_cache.sh /
# precompute_*_masks.sh RUN_ROOT=aig_train_run, and warmup_test_cache.sh only
# ever warms the test split under $EVAL_ROOT.
#
# For CPU inference numbers (RQ4 motivation: does inference stay cheap on
# CPU?) use test_cpu.sh instead — kept as a separate script on the genoa
# CPU-only partition so that comparison doesn't consume GPU budget.
#
# REQUIRED BEFORE RUNNING: the test split's graph cache and reduction masks
# must exist first (warmup_train_cache.sh only builds train+val). The exact
# submission chain lives in EVALUATION.md — follow it there rather than
# copying a second copy into this header, which is how the two drifted apart
# before. One thing that chain gets right and is easy to get wrong: masks
# must land in the same workspace this script reads ($EVAL_ROOT). (A prior
# version of this comment said SPARSIFICATION_ALGO defaults to and_gate_only
# and needs one submission PER method — that's now stale: it defaults to
# "all", one submission covers every method, see precompute_sparsification_
# masks.sh.)
# A missing mask raises at eval time rather than silently recomputing, so a
# mismatch fails the array task loudly instead of costing a slow rebuild.
#
# Once the warmup+mask chain completes, this array job only *reads* the
# cache — no concurrent-write contention regardless of parallelism, and it
# can run alongside test_cpu.sh.
# ---------------------------------------------------------------------------

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "=========================================="
echo "JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# =========================================================
# 1. Environment
# =========================================================

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
source "$VENV_PATH/bin/activate"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. ARRAY MAPPING — same "type:method" convention as benchmark.sh /
#    test_cpu.sh, so adding summarization later is one appended line here
#    (and in those two scripts) plus bumping --array's range.
# =========================================================

ALGORITHM="Orchestrate"
CONFIGS=(
    "none:none"
    "sparsification:and_gate_only" "sparsification:random_edge_dropout"
    "sparsification:spanning_forest" "sparsification:pagerank"
    "partition:random" "partition:metis"
    "partition:level_slicing" "partition:span_weighted_metis"
    # "summarization:<method>"   # appended here once implemented
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
REDUCTION_TYPE="${CONFIG%%:*}"
REDUCTION_METHOD="${CONFIG##*:}"

echo "Task $SLURM_ARRAY_TASK_ID assigned to reduction_type=$REDUCTION_TYPE reduction_method=$REDUCTION_METHOD"

CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

# =========================================================
# 3. PATHS — must match train.sh's checkpoint/cache layout exactly.
# =========================================================

# Checkpoints stay in the train workspace (that is where training wrote them).
# The eval CACHE is a separate workspace so test-split graphs never mix into
# the train cache — the design-level split makes train/test graphs disjoint,
# so nothing is shared anyway. Override the eval root with EVAL_ROOT=...
TRAIN_ROOT="/scratch-shared/$USER/aig_train_run"
CHECKPOINT_DIR="$TRAIN_ROOT/${ALGORITHM}/checkpoints"
EVAL_ROOT="${EVAL_ROOT:-/scratch-shared/$USER/aig_eval_run}"
CACHE_DIR="$EVAL_ROOT/${ALGORITHM}/cache"
TIER0_CACHE_DIR="$EVAL_ROOT/shared_tier0_cache"
TIER1_CACHE_DIR="$EVAL_ROOT/shared_tier1_cache"

# Validation-pass cache: the TRAIN workspace, where val's graphs and
# reduction masks actually live (see the header comment above). Sparsification
# masks under this root are found via config.SPARSIFICATION_REPLACE_PATH's
# aig_train_run -> aig_mask_cache redirect inside AIGGraphRegressionDataset;
# partition masks are written in-place here, so no redirect is needed.
VAL_CACHE_DIR="$TRAIN_ROOT/${ALGORITHM}/cache"
VAL_TIER0_CACHE_DIR="$TRAIN_ROOT/shared_tier0_cache"
VAL_TIER1_CACHE_DIR="$TRAIN_ROOT/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR/predictions"

NUM_WORKERS="${NUM_WORKERS:-12}"

# Batching is NOT set here on purpose. config.EVAL_DYNAMIC_BATCHING /
# config.EVAL_MAX_TOTAL_NODES_PER_BATCH are the single source of truth, so all
# 9 array tasks (and test_cpu.sh) cannot drift onto different batchings and
# make their hardware columns incomparable. The effective setting is recorded
# in each result row's `batching` column. Change it in config.py, not here.

# WandB run named test_<config>_<device>, mirroring train.py's train_<config>.
# WANDB_INIT_TIMEOUT matches train.sh: without it a compute node that cannot
# reach the wandb backend hangs in wandb.init() until the wall clock kills the
# task, with no results written.
export WANDB_INIT_TIMEOUT=120
WANDB="${WANDB:-true}"

# HARDCODED true, not an overridable env var: this is set for a specific
# targeted resubmission (fixing eval-workspace sparsification masks for
# array indices 2-4, whose full_graph pass already completed correctly on a
# prior run and only needs matched_reduction/val rerun — see EVALUATION.md/
# git history around commit c308b27).
#
# MUST BE REVERTED TO false BEFORE THE NEXT FULL SWEEP (array 0-8 on
# newly-trained checkpoints). For the 8 reduction configs this silently drops
# only the full_graph pass; for the BASELINE config (array index 0,
# "none:none") it is worse — build_eval_passes() only appends matched_reduction
# when reduction_type != "none", so baseline has no fallback pass at all and
# this would silently drop its ENTIRE test-split accuracy result, leaving
# only the val sanity-check row. Nothing in the job log flags this as an
# anomaly — only the absence of a "Running split='test' 'full_graph' pass"
# line reveals it, so a forgotten revert would look like a normal run.
SKIP_FULL_GRAPH=true
if [[ "$SKIP_FULL_GRAPH" == "true" ]]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "WARNING: SKIP_FULL_GRAPH is hardcoded true in this script — the"
    echo "full_graph pass will NOT run for task $SLURM_ARRAY_TASK_ID."
    echo "If this is array index 0 (baseline, none:none), NO test-split"
    echo "accuracy metrics will be produced at all this run (baseline has no"
    echo "matched_reduction fallback). Revert to SKIP_FULL_GRAPH=false in"
    echo "src/shell/test.sh before the next full 0-8 sweep."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

nvidia-smi -L

# =========================================================
# 4. EXECUTE
# =========================================================

REDUCTION_ARGS=()
if [[ "$REDUCTION_TYPE" != "none" ]]; then
    REDUCTION_ARGS=(--reduction_method "$REDUCTION_METHOD")
fi

# Pass-through for one-off flags, e.g. a diagnostic val-split run:
#   sbatch --export=ALL,EXTRA_ARGS="--split val --wandb false" src/shell/test.sh
# Word-split into an array so multiple flags work; the ${a[@]+...} guard keeps
# an unset/empty value safe under `set -u`.
EXTRA_ARGS="${EXTRA_ARGS:-}"
read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"

srun python -u -m test \
    --algorithm          "$ALGORITHM" \
    --reduction_type     "$REDUCTION_TYPE" \
    ${REDUCTION_ARGS[@]+"${REDUCTION_ARGS[@]}"} \
    --csv_paths          "$CSV_PATH" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --tier0_cache_dir    "$TIER0_CACHE_DIR" \
    --tier1_cache_dir    "$TIER1_CACHE_DIR" \
    --val_cache_dir       "$VAL_CACHE_DIR" \
    --val_tier0_cache_dir "$VAL_TIER0_CACHE_DIR" \
    --val_tier1_cache_dir "$VAL_TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --num_workers        "$NUM_WORKERS" \
    --wandb              "$WANDB" \
    --skip_full_graph    "$SKIP_FULL_GRAPH" \
    --device             cuda \
    --dump_predictions   true \
    --results_dir        "$RESULTS_DIR/inference_results" \
    --predictions_dir    "$RESULTS_DIR/predictions" \
    ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"}

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID GPU inference eval ($REDUCTION_TYPE/$REDUCTION_METHOD) complete."
echo "End time: $(date)"
