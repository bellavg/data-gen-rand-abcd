#!/bin/bash
#SBATCH --job-name=aig_opt
#SBATCH --time=24:00:00                  
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1                         
#SBATCH --mem=180G                       
#SBATCH --array=2-5                      # ONLY LAUNCH 2 TO 5
#SBATCH --output=logs/optuna_worker_%a.out 

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:-2}

SCRIPT_VERSION="2026-04-21 (Optuna Array Workers 2-5 - No Pip Install)"

echo "=========================================="
echo "JOB: Optuna Hyperparameter Tuning (Worker $TASK_ID)"
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

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"

STORAGE_PATH="$WORKSPACE/optuna_study.log"
export STORAGE_PATH

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# 5. Define Input CSVs
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER
# ---------------------------------------------------------
NUM_WORKERS=2 

echo "Starting Worker $TASK_ID on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "aig_opt_hp_tuning" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
    --log_dir "$LOG_DIR/worker_${TASK_ID}" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
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
echo "JOB complete."
echo "End time: $(date)"