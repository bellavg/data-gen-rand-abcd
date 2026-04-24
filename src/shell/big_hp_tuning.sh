#!/bin/bash
#SBATCH --job-name=big_optuna
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --array=1-3                      # ONLY LAUNCH 1 TO 3
#SBATCH --output=logs/big_optuna_worker_%a.out

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

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

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Skip pip installations since the venv is already set up!

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

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
NUM_WORKERS=0

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"

EXTRA_FLAGS=()
if [ "$PIN_MEMORY" = "true" ]; then
    EXTRA_FLAGS+=(--pin_memory)
fi
if [ "$PERSISTENT_WORKERS" = "true" ]; then
    EXTRA_FLAGS+=(--persistent_workers)
fi

echo "Starting Big Worker $TASK_ID on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "$STUDY_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
    --log_dir "$LOG_DIR/worker_${TASK_ID}" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    --train_samples 50000 \
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
