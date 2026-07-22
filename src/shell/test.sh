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
# non-baseline configs).
#
# For CPU inference numbers (RQ4 motivation: does inference stay cheap on
# CPU?) use test_cpu.sh instead — kept as a separate script on the genoa
# CPU-only partition so that comparison doesn't consume GPU budget.
#
# REQUIRED BEFORE RUNNING: the test split's graph cache and reduction masks
# must exist first (warmup_train_cache.sh only builds train+val). Chain:
#
#   W=$(sbatch --parsable src/shell/warmup_test_cache.sh)
#   S1=$(SPARSIFICATION_ALGO=and_gate_only       sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
#   S2=$(SPARSIFICATION_ALGO=random_edge_dropout sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
#   S3=$(SPARSIFICATION_ALGO=spanning_forest     sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
#   S4=$(SPARSIFICATION_ALGO=pagerank            sbatch --parsable --dependency=afterok:$W src/shell/precompute_sparsification_masks.sh)
#   M=$(sbatch --parsable --dependency=afterok:$W src/shell/precompute_partition_masks.sh)
#   sbatch --dependency=afterok:$S1:$S2:$S3:$S4:$M src/shell/test.sh
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

WORKSPACE="/scratch-shared/$USER/aig_train_run/${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
CACHE_DIR="$WORKSPACE/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR/predictions"

NUM_WORKERS="${NUM_WORKERS:-12}"

nvidia-smi -L

# =========================================================
# 4. EXECUTE
# =========================================================

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
    --device             cuda \
    --dump_predictions   true \
    --results_csv        "$RESULTS_DIR/inference_results.csv" \
    --predictions_dir    "$RESULTS_DIR/predictions"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID GPU inference eval ($REDUCTION_TYPE/$REDUCTION_METHOD) complete."
echo "End time: $(date)"
