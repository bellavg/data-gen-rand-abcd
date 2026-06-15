#!/bin/bash
#SBATCH --job-name=aig_train_array
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=0-3   
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_%A_%a.out    # %A is the array master job ID, %a is the task index
#
# Recommended: pre-warm the dataset cache on a CPU node before submitting this
# job so the GPU is not idle during graph loading.  Chain with warmup_cache.sh:
#
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/train.sh
#
# If the cache already exists (sentinel file present) the warmup script exits
# immediately, so the dependency is cheap.

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

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

# TCMalloc prevents glibc malloc fragmentation over long training runs.
module load gperftools/2.16-GCCcore-14.2.0
export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"

# Activate Virtual Environment
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"


# Verify WandB Authentication (Fail-fast check to prevent SLURM hangs)
echo "Checking Weights & Biases authentication..."
if ! python -c "import wandb; exit(0) if wandb.login(anonymous='never') else exit(1)" 2>/dev/null; then
    echo "CRITICAL ERROR: WandB is not authenticated! Run 'wandb login' in an active terminal before submitting this job." >&2
    exit 1
fi
echo "WandB authentication successful."

# Setup Paths
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$BASE_DIR"


# =========================================================
# 2. ARRAY MAPPING (Algorithm & Data)
# =========================================================

# Define our 3 algorithms
ALGORITHMS=("Orchestrate" "Deepsyn" "C2RS")

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
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$TIER0_CACHE_DIR"
mkdir -p "$TIER1_CACHE_DIR"

# Path to the hyperparameter tuning splits JSON file
# *************************************************************************
# USER REVIEW REQUIRED: Make sure this filename perfectly matches the one 
# generated inside your shared_cache during the hp_tuning run!
# *************************************************************************
HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
# All 50K graphs used across both HP tuning stages (15K Stage-1 + 35K Stage-2).
# Using this file ensures zero HP tuning leakage into final train/val/test splits.
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

if [ ! -f "$HP_TUNING_SPLITS" ]; then
    echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"
    echo "Please check the shared_cache directory to find the exact filename."
fi

# =========================================================
# 4. Runtime settings
# =========================================================
# Number of data-loader workers (default: SLURM_CPUS_PER_TASK or 16)
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
# =========================================================
# 5. EXECUTE TRAINING
# =========================================================

echo "Starting Final Training for $ALGORITHM on GPU 0..."

srun python -u -m train \
    --algorithm         "$ALGORITHM" \
    --csv_paths         "$CSV_PATH" \
    --checkpoint_dir    "$CHECKPOINT_DIR" \
    --log_dir           "$LOG_DIR" \
    --cache_dir         "$CACHE_DIR" \
    --tier0_cache_dir   "$TIER0_CACHE_DIR" \
    --tier1_cache_dir   "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --prefetch_factor   4 \
    --num_workers       8 \
    --patience          10

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID for $ALGORITHM complete."
echo "End time: $(date)"