#!/bin/bash
#SBATCH --job-name=precompute_partition_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_partition_%j.out

# ---------------------------------------------------------------------------
# Precompute dynamic-k partition masks for a cache workspace.
#
# Run this AFTER the matching cache warmup has finished, since it appends masks
# to the existing cached graph .pt files. RUN_ROOT selects the workspace and
# now defaults to the EVAL cache (see the RUN_ROOT block below) — so the bare
# invocation pairs with warmup_test_cache.sh, and building masks for TRAINING
# needs the train root passed explicitly.
#
# This script runs a single unified job to process all cache directories at
# once (including shared tier-0 and tier-1 directories). It automatically
# deduplicates .pt files to ensure each shared graph is only processed once.
#
# A workspace with no cached .pt files yet is SKIPPED per directory (exit 0,
# no masks written), so pointing this at the wrong/unwarmed root fails quietly
# — check the per-directory "Skipping" lines in the job log.
#
# Usage (eval workspace, the default — see EVALUATION.md for the full chain):
#   sbatch src/shell/precompute_partition_masks.sh
#
# Training workspace:
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   RUN_ROOT=/scratch-shared/$USER/aig_train_run \
#     sbatch --dependency=afterok:$WID src/shell/precompute_partition_masks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Define the algorithm argument (Python script supports "all" to compute them in one pass)
PARTITION_ALGO="all"

# Workspace root + algorithm. Defaults to the EVAL workspace (see
# EVALUATION.md) — masks for training were precomputed long ago, so eval is the
# live use case and the safe default. To (re)build masks for TRAINING you must
# pass the train root explicitly:
# RUN_ROOT=/scratch-shared/$USER/aig_train_run
# Partition masks are written in-place in the cache dirs (no redirect), so no
# MASK_CACHE_ROOT handling is needed here.
RUN_ROOT="${RUN_ROOT:-/scratch-shared/$USER/aig_eval_run}"
ALGORITHM="${ALGORITHM:-Orchestrate}"

echo "=========================================="
echo "PRECOMPUTE PARTITION MASKS JOB ID: $SLURM_JOB_ID"
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

# Only cover the cache roots that Orchestrate training can read from:
# - shared tier-0 cache
# - shared tier-1 cache
# - Orchestrate workspace cache (tier-2 / non-tiered graphs)
CACHE_DIRS=(
    "$RUN_ROOT/shared_tier0_cache"
    "$RUN_ROOT/shared_tier1_cache"
    "$RUN_ROOT/${ALGORITHM}/cache"
)

for SHARED_DIR in "${CACHE_DIRS[@]}"; do
    echo "=========================================="
    echo "Processing $SHARED_DIR"
    echo "=========================================="

    mkdir -p "$SHARED_DIR"

    if ! find "$SHARED_DIR" -maxdepth 1 -name '*.pt' -print -quit | grep -q .; then
        echo "No cached .pt graphs found in $SHARED_DIR yet. Skipping."
        continue
    fi
    
    echo "Running partition precomputation directly against shared cache..."
    time python -u -m data.partition "$PARTITION_ALGO" \
        --dirs "$SHARED_DIR" \
        --out-dirs "$SHARED_DIR"
done

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
