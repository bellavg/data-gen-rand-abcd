#!/bin/bash
#SBATCH --job-name=precompute_partition_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --array=0-3
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_partition_%A_%a.out

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

# Define our 4 partition methods
PARTITIONS=("metis" "span_weighted_metis" "level_slicing" "random")

# Select the partition algorithm for this specific array task
PARTITION_ALGO=${PARTITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "PRECOMPUTE PARTITION MASKS JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
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

# Shared caches for tier-0 and tier-1 graphs (Source)
SHARED_CACHES=(
    "/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
    "/scratch-shared/$USER/aig_train_run/shared_tier1_cache"
)

if [[ -z "${TMPDIR:-}" ]]; then
    export TMPDIR="/scratch-shared/$USER/tmp"
fi
mkdir -p "$TMPDIR"

LOCAL_WORKSPACE="$(mktemp -d "$TMPDIR/aig_cache_XXXXXX")"
echo "Creating local NVMe workspace at $LOCAL_WORKSPACE..."

# Ensure cleanup on exit
trap 'rm -rf "$LOCAL_WORKSPACE" && echo "Cleaned up $LOCAL_WORKSPACE"' EXIT

for SHARED_DIR in "${SHARED_CACHES[@]}"; do
    CACHE_NAME=$(basename "$SHARED_DIR")
    LOCAL_DIR="$LOCAL_WORKSPACE/$CACHE_NAME"
    
    echo "=========================================="
    echo "Processing $CACHE_NAME"
    echo "=========================================="
    
    echo "Copying $CACHE_NAME to local NVMe via tar pipe..."
    mkdir -p "$LOCAL_DIR"
    time tar -cf - -C "$SHARED_DIR" . | tar -xf - -C "$LOCAL_DIR"

    echo "Running partition precomputation against local NVMe directory..."
    # The python script will read data from LOCAL_DIR but save indices directly to SHARED_DIR
    time python -u -m data.partition "$PARTITION_ALGO" \
        --dirs "$LOCAL_DIR" \
        --out-dirs "$SHARED_DIR"

    echo "Cleaning up local workspace for $CACHE_NAME to save space..."
    rm -rf "$LOCAL_DIR"
done

echo "Cleaning up entire local workspace..."
rm -rf "$LOCAL_WORKSPACE"

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
