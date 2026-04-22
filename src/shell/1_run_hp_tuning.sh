#!/bin/bash
#SBATCH --job-name=aig_optuna
#SBATCH --time=24:00:00          # 5 days
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1       
#SBATCH --output=logs/optuna_%j.out

set -euo pipefail

SCRIPT_VERSION="2026-04-21 (Single GPU Update)"

echo "=========================================="
echo "JOB: Optuna Hyperparameter Tuning (5 Days)"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Script version: $SCRIPT_VERSION"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

if [ ! -d "/scratch-shared/$USER/.venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv /scratch-shared/$USER/.venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

# 2. Activate Virtual Environment
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
echo "Using BASE_DIR=$BASE_DIR"
echo "PYTHONPATH=$PYTHONPATH"
echo "python: $(command -v python)"
echo "pip:    $(command -v pip)"
run_clean python --version

cd "$BASE_DIR"

# 3. Environment sanity check (no pip installs)
echo "Assuming required Python packages and prebuilt PyG extensions are already installed in the virtualenv."
if ! run_clean python -c "import torch, optuna, pytorch_lightning, torch_geometric" >/dev/null 2>&1; then
    echo "Required Python packages not found in venv. Please install them before running this script."
    echo "Recommended command (run once): pip install -e '.[dev]' --no-deps"
    exit 1
fi

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"

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
# NEW 1-GPU LOGIC
# ---------------------------------------------------------
WORKER_COUNT=1
echo "Launching $WORKER_COUNT Optuna worker process."

# If OOM hangs continue to happen, set this to 0. Otherwise, 8 is faster.
NUM_WORKERS=4
echo "Using num_workers per process: $NUM_WORKERS"

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

EXTRA_FLAGS=()
if [ "$PIN_MEMORY" = "true" ]; then
    EXTRA_FLAGS+=(--pin_memory)
fi
if [ "$PERSISTENT_WORKERS" = "true" ]; then
    EXTRA_FLAGS+=(--persistent_workers)
fi

# 6. Launch Worker 0 (Pinned to GPU 0)
echo "Starting Worker 0 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "aig_opt_hp_tuning" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
    --log_dir "$LOG_DIR/worker_0" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    "${EXTRA_FLAGS[@]}" \
    > "$LOG_DIR/worker_0.log" 2>&1 &
PID0=$!

# 7. Wait for worker to complete
FAIL=0
if ! wait "$PID0"; then
    echo "Worker 0 failed. Last 80 lines:"
    tail -n 80 "$LOG_DIR/worker_0.log" || true
    FAIL=1
fi

if (( FAIL != 0 )); then
    echo "The worker failed; exiting non-zero."
    exit 1
fi

echo "=========================================="
echo "JOB complete."
echo "End time: $(date)"