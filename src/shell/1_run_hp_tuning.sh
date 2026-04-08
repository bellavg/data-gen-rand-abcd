#!/bin/bash
#SBATCH --job-name=aig_optuna
#SBATCH --time=120:00:00          # 5 days
#SBATCH --nodes=1                 # Single node to ensure SQLite works correctly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # 16 cores per H100 GPU worker (instead of 36)
#SBATCH --partition=gpu_h100      # Use the H100 partition      # 18 cores per GPU worker
#SBATCH --gpus=2                  # 2 GPUs
#SBATCH --output=logs/optuna_%j.out


set -euo pipefail

echo "=========================================="
echo "JOB: Optuna Hyperparameter Tuning (5 Days)"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# 2. Activate Virtual Environment
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure the project src directory is on PYTHONPATH so `python -m hp_tuning`
# can find `hp_tuning.py` when run from the job working directory.
# Use an absolute BASE_DIR pointing at the repo in the user's home directory.
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
export PYTHONPATH="$BASE_DIR/src:${PYTHONPATH:-}"
echo "Using BASE_DIR=$BASE_DIR"
echo "PYTHONPATH=$PYTHONPATH"

# 3. Install Local Library (avoid building PyG native extensions in isolated build env)
# Note: Ensure you run `sbatch run_optuna.sh` from the root of your project
# where your pyproject.toml is located.
echo "Preparing environment and installing prebuilt PyG extensions (torch-scatter)..."
# Upgrade packaging/build tools
pip install --upgrade pip setuptools wheel

# Ensure `torch` is installed in the virtualenv (install a matching wheel if needed)
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Torch not found in venv — installing torch (adjust version as needed)..."
    # Adjust the torch version to match your cluster's CUDA/arch (example below)
    pip install torch==2.11.0
fi

# Install prebuilt torch-scatter wheel from PyG index (no deps). Choose the correct
# wheel for your torch+CUDA combination at https://data.pyg.org/whl/ and adjust as needed.
echo "Installing torch-scatter from PyG wheels (no deps)..."
pip install --no-deps -f https://data.pyg.org/whl/ torch-scatter

# Install local package without dependencies because we've handled them manually
pip install -e '.[dev]' --no-deps

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
DB_PATH="$WORKSPACE/optuna_study.db"
DB_URL="sqlite:///${DB_PATH}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "Using Database: $DB_URL"
# 5. Define Input CSVs (use BASE_DIR for portability)
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

echo "Using Database: $DB_URL"

# Compute number of DataLoader workers to pass to each worker process.
# Default: split available CPUs between the two worker processes.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))
else
    NUM_WORKERS=${NUM_WORKERS:-16}
fi
# Optionally reserve one CPU for OS/overhead by uncommenting the next line
# NUM_WORKERS=$((NUM_WORKERS - 1))
echo "Using num_workers per process: $NUM_WORKERS"

# 6. Launch Worker 1 (Pinned to GPU 0)
echo "Starting Worker 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$DB_URL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    > "$LOG_DIR/worker_0.log" 2>&1 &

# Sleep briefly to ensure Worker 1 initializes the SQLite file before Worker 2 tries to read it
sleep 15 

# 7. Launch Worker 2 (Pinned to GPU 1)
echo "Starting Worker 2 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m hp_tuning \
    --db_url "$DB_URL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    > "$LOG_DIR/worker_1.log" 2>&1 &

# 8. Wait for workers to complete or hit the 5-day walltime
wait

echo "=========================================="
echo "JOB complete."
echo "End time: $(date)"