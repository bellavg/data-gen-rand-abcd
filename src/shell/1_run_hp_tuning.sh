#!/bin/bash
#SBATCH --job-name=aig_optuna
#SBATCH --time=120:00:00          # 5 days
#SBATCH --nodes=1                 # Single node to ensure SQLite works correctly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36        # 18 cores per GPU worker
#SBATCH --gpus=2                  # 2 GPUs
#SBATCH --partition=gpu
#SBATCH --output=/scratch-shared/%u/optuna_master_%j.out
#SBATCH --error=/scratch-shared/%u/optuna_master_%j.err

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

# 3. Install Local Library
# Note: Ensure you run `sbatch run_optuna.sh` from the root of your project
# where your setup.py or pyproject.toml is located.
echo "Installing project in editable mode..."
pip install -e .

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
DB_PATH="$WORKSPACE/optuna_study.db"
DB_URL="sqlite:///${DB_PATH}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# 5. Define Input CSVs
CSV_1="/home/$USER/data-gen-rand-abcd/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="/home/$USER/data-gen-rand-abcd/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="/home/$USER/data-gen-rand-abcd/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="/home/$USER/data-gen-rand-abcd/data/designs/design_metadata/algo_C2RS_ml.csv"

echo "Using Database: $DB_URL"

# 6. Launch Worker 1 (Pinned to GPU 0)
echo "Starting Worker 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$DB_URL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    > "$LOG_DIR/worker_0.log" 2>&1 &

# Sleep briefly to ensure Worker 1 initializes the SQLite file before Worker 2 tries to read it
sleep 15 

# 7. Launch Worker 2 (Pinned to GPU 1)
echo "Starting Worker 2 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m hp_tuning \
    --db_url "$DB_URL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    > "$LOG_DIR/worker_1.log" 2>&1 &

# 8. Wait for workers to complete or hit the 5-day walltime
wait

echo "=========================================="
echo "JOB complete."
echo "End time: $(date)"