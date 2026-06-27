#!/bin/bash
#SBATCH --job-name=precompute_levels
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_levels_%j.out

# ---------------------------------------------------------------------------
# Precompute node levels for training caches.
#
# Usage:
#   sbatch src/shell/precompute_levels.sh
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "PRECOMPUTE LEVELS JOB ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "=========================================="

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

# Prevent PyTorch from using all cores per worker process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# Shared caches for tier-0 and tier-1 graphs (Source)
SHARED_CACHES=(
    "/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
    "/scratch-shared/$USER/aig_train_run/shared_tier1_cache"
)

for SHARED_DIR in "${SHARED_CACHES[@]}"; do
    echo "=========================================="
    echo "Processing $SHARED_DIR"
    echo "=========================================="
    
    echo "Running levels precomputation directly against shared cache..."
    time python -u -m data.compute_levels \
        --dirs "$SHARED_DIR" \
        --out-dirs "$SHARED_DIR"

echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
