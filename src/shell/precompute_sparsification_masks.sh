#!/bin/bash
#SBATCH --job-name=precompute_sparsification_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --array=0-2
#SBATCH --output=logs/precompute_sparsification_%A_%a.out

# ---------------------------------------------------------------------------
# Precompute sparsification masks for training caches.
#
# Runs AFTER warmup_train_cache.sh has populated the cache directories.
# Writes one index file per (cache directory, algorithm):
#   {cache_dir}/_sparse_{algo}.pt
# No individual graph .pt files are modified.
#
# and_gate_only is NOT precomputed here — it is applied on-the-fly in
# dataset.get() since it is a fast deterministic transform (~1-5 ms/graph).
#
# Array tasks:
#   0 = random_edge_dropout
#   1 = spanner
#   2 = pagerank
#
# Usage:
#   sbatch src/shell/precompute_sparsification_masks.sh
#
# Or chain after a successful warmup job:
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/precompute_sparsification_masks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SPARSIFICATION_ALGOS=("random_edge_dropout" "spanner" "pagerank")
SPARSIFICATION_ALGO=${SPARSIFICATION_ALGOS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "PRECOMPUTE SPARSIFICATION MASKS JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "Sparsification algorithm: ${SPARSIFICATION_ALGO}"
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

# Prevent PyTorch from using all cores per worker process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. CACHE DIRECTORIES
# =========================================================

TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# =========================================================
# 3. EXECUTE PIPELINE
# =========================================================

echo "Running sparsification precomputation for all cache directories..."
python -u -m data.sparsification "$SPARSIFICATION_ALGO" \
    --dirs \
        "$TIER0_CACHE_DIR" \
        "$TIER1_CACHE_DIR"

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
