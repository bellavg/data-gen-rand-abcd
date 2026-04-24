#!/bin/bash
#SBATCH --job-name=aig_train_array
#SBATCH --time=72:00:00                  
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=0-3                      # 4 jobs total (indices 0, 1, 2, 3)
#SBATCH --output=logs/train_%A_%a.out    # %A is the array master job ID, %a is the task index

set -euo pipefail

echo "=========================================="
echo "JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# Activate Virtual Environment
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Setup Paths
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

cd "$BASE_DIR"

# =========================================================
# 2. ARRAY MAPPING (Algorithm & Data)
# =========================================================

# Define our 4 algorithms
ALGORITHMS=("Orchestrate" "Deepsyn" "Syn4" "C2RS")

# Select the algorithm for this specific array task
ALGORITHM=${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}

# Set the CSV path dynamically based on the chosen algorithm
CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

echo "Task $SLURM_ARRAY_TASK_ID assigned to ALGORITHM: $ALGORITHM"
echo "Using CSV dataset: $CSV_PATH"

# =========================================================
# 3. DEFINE OUTPUT & CACHE PATHS
# =========================================================

# Workspace for the current training runs (Separated by Algorithm)
WORKSPACE="/scratch-shared/$USER/aig_train_run/${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
CACHE_DIR="$WORKSPACE/cache"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

# Path to the hyperparameter tuning splits JSON file
# *************************************************************************
# USER REVIEW REQUIRED: Make sure this filename perfectly matches the one 
# generated inside your shared_cache during the hp_tuning run!
# *************************************************************************
HP_TUNING_WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
# Assuming you ran tuning on all 4 CSVs and didn't set a sample limit, it might look like this:
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/hp_tuning/shared_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_20000_splits.json"

if [ ! -f "$HP_TUNING_SPLITS" ]; then
    echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"
    echo "Please check the shared_cache directory to find the exact filename."
fi

# =========================================================
# 4. USER CONFIGURATION (Model Hyperparameters)
# =========================================================


MAX_EPOCHS=100
NUM_WORKERS=16 # Matches SLURM_CPUS_PER_TASK

# =========================================================
# 5. EXECUTE TRAINING
# =========================================================

echo "Starting Final Training for $ALGORITHM on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --algorithm "$ALGORITHM" \
    --csv_paths "$CSV_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --cache_dir "$CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --encoder_name "$ENCODER_NAME" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --check_val_every_n 3 \
    --patience 10 \
    --scheduler_patience 4

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID for $ALGORITHM complete."
echo "End time: $(date)"