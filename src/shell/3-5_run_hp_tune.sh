#!/bin/bash
#SBATCH --job-name=aig_opt_3_to_5
#SBATCH --time=24:00:00                  
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1                         
#SBATCH --mem=180G                       # EXPLICITLY REQUEST 180 GB RAM
#SBATCH --array=3-5                      # THIS LAUNCHES JOBS 3, 4, and 5
#SBATCH --output=logs/optuna_worker_%a.out # '%a' automatically inserts the array ID

set -euo pipefail

# Grab the unique task ID assigned by SLURM (defaults to 3 if run manually)
TASK_ID=${SLURM_ARRAY_TASK_ID:-3}

SCRIPT_VERSION="2026-04-21 (Optuna Array Workers 3-5)"

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

run_clean() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$@"
}

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

cd "$BASE_DIR"

# 3. Install Local Library (Skipping full text output for brevity, but keeping logic)
run_clean pip install --upgrade pip "setuptools<82" wheel >/dev/null 2>&1
run_clean pip install -e '.[dev]' --no-deps >/dev/null 2>&1

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"

# KEEP THIS THE SAME SO THEY JOIN WORKERS 1 & 2
STORAGE_PATH="$WORKSPACE/optuna_study.log"
export STORAGE_PATH

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "Using JournalStorage at: $STORAGE_PATH"

# 5. Define Input CSVs
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER
# ---------------------------------------------------------
# Keep NUM_WORKERS at 2 to avoid OOM
NUM_WORKERS=2 
echo "Using num_workers per process: $NUM_WORKERS"

echo "Starting Worker $TASK_ID on GPU 0..."

# We use TASK_ID to ensure logs go to different folders/files
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