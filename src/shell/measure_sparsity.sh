#!/bin/bash
#SBATCH --job-name=measure_sparsity
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/measure_sparsity_%j.out

# ===========================================================================
# measure_sparsity.sh
#
# Measures the actual edge/node retention of each sparsification method on
# a sample of graphs.  Reports per-method statistics so you can calibrate
# parameters (dropout rate, spanner stretch, pagerank keep_ratio) to achieve
# comparable sparsity levels across methods.
#
# Two modes:
#   1. --mask-cache-dirs  Scan precomputed _sparse_*.pt index files (fast).
#   2. --graph-dir        Load .pt graph files and compute masks on-the-fly.
#
# Submit with:
#   sbatch src/shell/measure_sparsity.sh
#
# Override the cache root or graph directory:
#   GRAPH_DIR=/path/to/graphs sbatch src/shell/measure_sparsity.sh
#   MASK_CACHE_DIR=/path/to/masks sbatch src/shell/measure_sparsity.sh
# ===========================================================================

set -euo pipefail

echo "=========================================="
echo "MEASURE SPARSIFICATION RETENTION"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# 1. Environment
if command -v module > /dev/null 2>&1; then
    module purge || true
    module load 2025 || true
    module load foss/2025a || true
    module load Python/3.13.1-GCCcore-14.2.0 || true
fi

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

# 2. Paths
MASK_CACHE_DIR="${MASK_CACHE_DIR:-/scratch-shared/$USER/aig_mask_cache}"
GRAPH_DIR="${GRAPH_DIR:-/scratch-shared/$USER/aig_train_run/shared_tier0_cache}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"

echo "BASE_DIR=$BASE_DIR"
echo "MASK_CACHE_DIR=$MASK_CACHE_DIR"
echo "GRAPH_DIR=$GRAPH_DIR"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "PYTHON_BIN=$PYTHON_BIN"
echo ""

# 3. Run measurement
# Force on-the-fly measurement from raw graph files to get accurate timing
# and to ensure we test the currently implemented algorithms.
if [[ -d "$GRAPH_DIR" ]]; then
    echo "Mode: computing masks on-the-fly from graph files in $GRAPH_DIR"
    "$PYTHON_BIN" -u "$BASE_DIR/src/data/measure_sparsity.py" \
        --graph-dir "$GRAPH_DIR" \
        --max-samples "$MAX_SAMPLES"
else
    echo "ERROR: GRAPH_DIR ($GRAPH_DIR) does not exist." >&2
    echo "Set it to a valid path containing .pt graph files and resubmit." >&2
    exit 1
fi

echo "=========================================="
echo "Measurement finished: $(date)"
echo "=========================================="
