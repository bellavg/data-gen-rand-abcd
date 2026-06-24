#!/bin/bash
#SBATCH --job-name=precompute_partition_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/precompute_partition_%J.out

# ---------------------------------------------------------------------------
# Precompute dynamic-k partition masks for training caches.
#
# Run this AFTER warmup_train_cache.sh has finished, since it appends masks
# to the existing cached graph .pt files.
#
# This script runs a single unified job to process all cache directories at
# once (including shared tier-0 and tier-1 directories). It automatically
# deduplicates .pt files to ensure each shared graph is only processed once.
#
# Usage:
#   sbatch src/shell/precompute_partition_masks.sh
#
# Or chain it to run automatically after a successful warmup job:
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/precompute_partition_masks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Partition algorithm to compute (choices: metis, span_weighted_metis, level_slicing, random, all)
# By default, we run 'all' to precompute all partition masks in one go.
PARTITION_ALGO="${PARTITION_ALGO:-all}"

echo "=========================================="
echo "PRECOMPUTE PARTITION MASKS JOB"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "Partition algorithm: ${PARTITION_ALGO}"
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

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
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

# Shared caches for tier-0 and tier-1 graphs (must match warmup_train_cache.sh)
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# Target algorithm cache directories
ORCHESTRATE_CACHE_DIR="/scratch-shared/$USER/aig_train_run/Orchestrate/cache"
DEEPSYN_CACHE_DIR="/scratch-shared/$USER/aig_train_run/Deepsyn/cache"
C2RS_CACHE_DIR="/scratch-shared/$USER/aig_train_run/C2RS/cache"

# =========================================================
# 3. EXECUTE PIPELINE
# =========================================================

echo "Running partition precomputation for all cache directories..."
python -u -m data.partition "$PARTITION_ALGO" \
    --dirs \
        "$TIER0_CACHE_DIR" \
        "$TIER1_CACHE_DIR" \
        "$ORCHESTRATE_CACHE_DIR" \
        "$DEEPSYN_CACHE_DIR" \
        "$C2RS_CACHE_DIR"

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
