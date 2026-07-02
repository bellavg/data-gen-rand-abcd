#!/bin/bash
#SBATCH --job-name=precompute_sparsification_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_sparsification_%j.out

# ---------------------------------------------------------------------------
# Precompute sparsification masks for training caches.
#
# Run this AFTER warmup_train_cache.sh has finished, since it reads the
# existing cached graph .pt files and writes sidecar index files.
#
# This script runs a single unified job to process all cache directories at
# once (including shared tier-0 and tier-1 directories). It automatically
# deduplicates .pt files to ensure each shared graph is only processed once.
#
# and_gate_only is NOT precomputed here — it is applied on-the-fly in
# dataset.get() since it is a fast deterministic transform (~1-5 ms/graph).
#
# Usage:
#   sbatch src/shell/precompute_sparsification_masks.sh
#
# Or chain it to run automatically after a successful warmup job:
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/precompute_sparsification_masks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Define the algorithm argument (Python script supports "all" to compute them in one pass)
SPARSIFICATION_ALGO="all"

echo "=========================================="
echo "PRECOMPUTE SPARSIFICATION MASKS JOB ID: $SLURM_JOB_ID"
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

BASE_DIR="$HOME/data-gen-sparsification"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

# Prevent PyTorch from using all 48 cores per worker process
# which causes severe thread thrashing and massive slowdowns.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. CACHE DIRECTORIES
# =========================================================

# Shared caches for tier-0 and tier-1 graphs (Source)
SHARED_CACHES=(
    "/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
    "/scratch-shared/$USER/aig_train_run/shared_tier1_cache"
)

for SHARED_DIR in "${SHARED_CACHES[@]}"; do
    echo "=========================================="
    echo "Processing $SHARED_DIR"
    echo "=========================================="

    echo "Running sparsification precomputation directly against shared cache..."
    time python -u -m data.sparsification "$SPARSIFICATION_ALGO" \
        --dirs "$SHARED_DIR" \
        --out-dirs "$SHARED_DIR"
done

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
