#!/bin/bash
#SBATCH --job-name=aig_test_cpu_array
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --array=0-8
#SBATCH --output=logs/test_cpu_%A_%a.out

# ---------------------------------------------------------------------------
# CPU inference eval — same accuracy + hardware sweep as test.sh, but on the
# CPU-only genoa partition with --device cpu. Kept as a SEPARATE script on a
# separate SLURM partition specifically so the CPU-vs-GPU inference
# comparison (RQ4 motivation: "inference on cpu — less intensive") doesn't
# spend GPU-node budget. Predictions are numerically identical to the GPU
# pass (only timing differs), so this runs with --dump_predictions false to
# avoid redundant writes — test.sh already wrote the per-graph CSVs. For the
# same reason this passes --skip_val true: test.sh's GPU run already covers
# the validation-set sanity pass, and CPU accuracy numbers on val add nothing
# this script cares about (it only measures inference hardware).
#
# Depends on the same warmup+mask chain as test.sh (see EVALUATION.md) and can
# run concurrently with test.sh — both are read-only against the cache once
# that chain has completed.
# ---------------------------------------------------------------------------

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "=========================================="
echo "JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# Same "type:method[:split_by]" array convention as test.sh / benchmark.sh.
# Indices 9-10 are opt-in: the baseline checkpoint (none:none) re-evaluated on
# the recipe/random splits instead of design — see test.sh's ARRAY MAPPING
# comment for why this isn't a full grid with the 8 reduction configs above.
ALGORITHM="Orchestrate"
CONFIGS=(
    "none:none"
    "sparsification:and_gate_only" "sparsification:random_edge_dropout"
    "sparsification:spanning_forest" "sparsification:pagerank"
    "partition:random" "partition:metis"
    "partition:level_slicing" "partition:span_weighted_metis"
    # "summarization:<method>"   # appended here once implemented
    "none:none:recipe" "none:none:random"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r REDUCTION_TYPE REDUCTION_METHOD SPLIT_BY <<< "$CONFIG"

echo "Task $SLURM_ARRAY_TASK_ID assigned to reduction_type=$REDUCTION_TYPE reduction_method=$REDUCTION_METHOD split_by=${SPLIT_BY:-<config default>}"

CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

# Checkpoints stay in the train workspace; the eval CACHE is separate (see
# test.sh). Override the eval root with EVAL_ROOT=...
TRAIN_ROOT="/scratch-shared/$USER/aig_train_run"
CHECKPOINT_DIR="$TRAIN_ROOT/${ALGORITHM}/checkpoints"
EVAL_ROOT="${EVAL_ROOT:-/scratch-shared/$USER/aig_eval_run}"
CACHE_DIR="$EVAL_ROOT/${ALGORITHM}/cache"
TIER0_CACHE_DIR="$EVAL_ROOT/shared_tier0_cache"
TIER1_CACHE_DIR="$EVAL_ROOT/shared_tier1_cache"

# Wired for parity with test.sh in case --skip_val is ever overridden false
# via EXTRA_ARGS below; --skip_val true (default here) makes these unused.
VAL_CACHE_DIR="$TRAIN_ROOT/${ALGORITHM}/cache"
VAL_TIER0_CACHE_DIR="$TRAIN_ROOT/shared_tier0_cache"
VAL_TIER1_CACHE_DIR="$TRAIN_ROOT/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

NUM_WORKERS="${NUM_WORKERS:-16}"

# SPLIT_BY comes from the array index decoded above — same convention as
# test.sh (empty for indices 0-8, meaning config.SPLIT_BY's default).
SPLIT_BY_ARGS=()
if [[ -n "${SPLIT_BY:-}" ]]; then
    SPLIT_BY_ARGS=(--split_by "$SPLIT_BY")
    echo "Using SPLIT_BY=$SPLIT_BY."
fi

# Batching comes from config.EVAL_* (see test.sh) so this and the GPU pass
# cannot end up comparing two different batchings rather than two devices.

# See test.sh: caps the wandb handshake so an unreachable backend cannot
# consume the whole array's wall clock.
export WANDB_INIT_TIMEOUT=120
WANDB="${WANDB:-true}"

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
    ${SPLIT_BY_ARGS[@]+"${SPLIT_BY_ARGS[@]}"} \
    --num_workers        "$NUM_WORKERS" \
    --wandb              "$WANDB" \
    --device             cpu \
    --dump_predictions   false \
    --skip_val           true \
    --results_dir        "$RESULTS_DIR/inference_results" \
    --predictions_dir    "$RESULTS_DIR/predictions" \
    ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"}

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID CPU inference eval ($REDUCTION_TYPE/$REDUCTION_METHOD, split_by=${SPLIT_BY:-<config default>}) complete."
echo "End time: $(date)"
