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
# avoid redundant writes — test.sh already wrote the per-graph CSVs.
#
# Depends on the same warmup+mask chain as test.sh (see its header comment)
# and can run concurrently with test.sh — both are read-only against the
# cache once that chain has completed.
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

# Same "type:method" array convention as test.sh / benchmark.sh.
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

WORKSPACE="/scratch-shared/$USER/aig_train_run/${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
CACHE_DIR="$WORKSPACE/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

NUM_WORKERS="${NUM_WORKERS:-16}"

REDUCTION_ARGS=()
if [[ "$REDUCTION_TYPE" != "none" ]]; then
    REDUCTION_ARGS=(--reduction_method "$REDUCTION_METHOD")
fi

srun python -u -m test \
    --algorithm          "$ALGORITHM" \
    --reduction_type     "$REDUCTION_TYPE" \
    ${REDUCTION_ARGS[@]+"${REDUCTION_ARGS[@]}"} \
    --csv_paths          "$CSV_PATH" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --tier0_cache_dir    "$TIER0_CACHE_DIR" \
    --tier1_cache_dir    "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --num_workers        "$NUM_WORKERS" \
    --device             cpu \
    --dump_predictions   false \
    --results_csv        "$RESULTS_DIR/inference_results.csv" \
    --predictions_dir    "$RESULTS_DIR/predictions"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID CPU inference eval ($REDUCTION_TYPE/$REDUCTION_METHOD) complete."
echo "End time: $(date)"
