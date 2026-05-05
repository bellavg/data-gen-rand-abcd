#!/bin/bash
#SBATCH --job-name=big_optuna
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH --output=logs/big_optuna_worker_1.out

set -euo pipefail

# No array: always worker 1
TASK_ID=1

SCRIPT_VERSION="2026-04-24 (Big Optuna Array Workers 1-3 - No Pip Install)"

echo "=========================================="
echo "JOB: Big Optuna Hyperparameter Tuning (Worker $TASK_ID)"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Script version: $SCRIPT_VERSION"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# --- Load gperftools TCMalloc to prevent glibc malloc fragmentation.
# TCMalloc aggressively returns freed memory to the OS so that thousands of
# sequential torch.load() calls over a long tuning run do not silently balloon
# the process RSS until SLURM kills it.
module load gperftools/2.16-GCCcore-14.2.0
export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "TCMalloc preloaded from: ${EBROOTGPERFTOOLS}/lib/libtcmalloc.so"
# -------------------------------------------------------------------------

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Skip pip installations since the venv is already set up!

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$BASE_DIR"

# 4. Define Output Paths in Scratch (separate big-run workspace)
WORKSPACE="/scratch-shared/$USER/big_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"

STORAGE_PATH="$WORKSPACE/optuna_study.log"
export STORAGE_PATH

STUDY_NAME="${STUDY_NAME:-big_optuna_hp_tuning}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# 5. Define Input CSVs
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER (big run)
# ---------------------------------------------------------
# Default to small parallelism to reduce main-process fragmentation from long
# single-process runs, while keeping host-memory queue depth constrained.
NUM_WORKERS="${NUM_WORKERS:-2}"

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
MEMORY_TELEMETRY_TRIALS="${MEMORY_TELEMETRY_TRIALS:-0}"

if [ "$NUM_WORKERS" -gt 0 ]; then
    if ! [[ "$PREFETCH_FACTOR" =~ ^[0-9]+$ ]]; then
        echo "Invalid PREFETCH_FACTOR='$PREFETCH_FACTOR'; defaulting to 1."
        PREFETCH_FACTOR=1
    fi

    if [ "$PREFETCH_FACTOR" -ne 1 ]; then
        echo "Clamping PREFETCH_FACTOR from $PREFETCH_FACTOR to 1 for OOM safety."
        PREFETCH_FACTOR=1
    fi
fi

if [ "$NUM_WORKERS" -eq 0 ]; then
    PERSISTENT_WORKERS=false
fi

EXTRA_FLAGS=()
if [ "$PIN_MEMORY" = "true" ]; then
    EXTRA_FLAGS+=(--pin_memory)
fi
if [ "$PERSISTENT_WORKERS" = "true" ]; then
    EXTRA_FLAGS+=(--persistent_workers)
fi

if [ "$NUM_WORKERS" -gt 0 ]; then
    EXTRA_FLAGS+=(--prefetch_factor "$PREFETCH_FACTOR")
fi
if [ "$DYNAMIC_BATCHING" = "true" ]; then
    EXTRA_FLAGS+=(--dynamic_batching)
fi

echo "DataLoader config: num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY persistent_workers=$PERSISTENT_WORKERS prefetch_factor=$PREFETCH_FACTOR dynamic_batching=$DYNAMIC_BATCHING"
echo "Memory telemetry: first $MEMORY_TELEMETRY_TRIALS trial(s)"

echo "Starting Big Worker $TASK_ID on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -u -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "$STUDY_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
    --log_dir "$LOG_DIR/worker_${TASK_ID}" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    --memory_guard_max_tokens 3.5e8 \
    --hard_prune_risk 120000 \
    --dataset_seed 42 \
    --memory_telemetry_trials "$MEMORY_TELEMETRY_TRIALS" \
    --train_samples 20000 \
    > "$LOG_DIR/worker_${TASK_ID}.log" 2>&1 &
PID=$!

FAIL=0
if ! wait "$PID"; then
    echo "Worker $TASK_ID failed. Last 80 lines:"
    tail -n 80 "$LOG_DIR/worker_${TASK_ID}.log" || true
    FAIL=1
fi

if (( FAIL != 0 )); then
    echo "Worker $TASK_ID failed; exiting non-zero."
    exit 1
fi

echo "=========================================="
echo "BIG JOB complete."
echo "End time: $(date)"
