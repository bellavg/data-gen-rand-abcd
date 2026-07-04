#!/bin/bash
#SBATCH --job-name=precompute_sparsification_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_sparsification_%j.out

# ---------------------------------------------------------------------------
# Precompute sparsification masks using Cache Manifests
# ---------------------------------------------------------------------------

set -euo pipefail

# Define the algorithm argument
SPARSIFICATION_ALGO="all"

# Match the workspace targeted in train.sh
ALGORITHM="${ALGORITHM:-Orchestrate}"

echo "=========================================="
echo "PRECOMPUTE SPARSIFICATION MASKS JOB ID: $SLURM_JOB_ID"
echo "Algorithm: $ALGORITHM"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "=========================================="

# =========================================================
# 1. Environment
# =========================================================

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
source "$VENV_PATH/bin/activate"

# BASE_DIR matches the repo root, just like in training
BASE_DIR="${BASE_DIR:-$HOME/data-gen-sparsification}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. MANIFEST DIRECTORIES
# =========================================================

MANIFEST_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache/metadata"

echo "=========================================="
echo "Processing manifests in $MANIFEST_DIR"
echo "=========================================="

# Call the updated python script with --manifest-dirs
time python -W ignore -u -m data.sparsification "$SPARSIFICATION_ALGO" \
    --manifest-dirs "$MANIFEST_DIR" \
    --replace-path "/scratch-shared/$USER/aig_train_run" "/scratch-shared/$USER/aig_mask_cache"

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
