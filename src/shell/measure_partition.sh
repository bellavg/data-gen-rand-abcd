#!/bin/bash
#SBATCH --job-name=measure_partition
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/measure_partition_%j.out

# ===========================================================================
# measure_partition.sh
#
# Partition twin of measure_sparsity.sh. Measures the offline cost of the
# partitioning algorithms themselves: edge-cut ratio, dynamic k, partition
# balance, and per-graph wall-clock time.
#
# Only one mode, unlike measure_sparsity.sh: partitioning has no precomputed
# _sparse_*.pt mask index to scan, so assignments are always computed
# on-the-fly from the .pt graph files.
#
# MAX_SAMPLES defaults to 10000 to match measure_sparsity.sh — the RQ2 offline
# table reports both reduction families side by side, and a 100-graph partition
# row next to a 10,000-graph sparsification row is not a comparison.
#
# Submit with:
#   sbatch src/shell/measure_partition.sh
#
# Override the sample count or graph tiers:
#   sbatch --export=ALL,MAX_SAMPLES=1000 src/shell/measure_partition.sh
#   sbatch --export=ALL,RUN_ROOT=/scratch-shared/$USER/aig_eval_run src/shell/measure_partition.sh
#
# Writes logs/partition_stats_{random,metis,level_slicing,span_weighted_metis}.csv
# relative to the submission directory, so submit from the repo root.
# ===========================================================================

set -euo pipefail

echo "=========================================="
echo "MEASURE PARTITIONING COST"
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
RUN_ROOT="${RUN_ROOT:-/scratch-shared/$USER/aig_train_run}"
TIER0_DIR="$RUN_ROOT/shared_tier0_cache"
TIER1_DIR="$RUN_ROOT/shared_tier1_cache"

echo "TIER0_DIR=$TIER0_DIR"
echo "TIER1_DIR=$TIER1_DIR"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "PYTHON_BIN=$PYTHON_BIN"
echo ""

# 3. Run measurement
if [[ -d "$TIER0_DIR" || -d "$TIER1_DIR" ]]; then
    echo "Mode: computing partition assignments on-the-fly from graph files across multiple tiers"
    "$PYTHON_BIN" -u "$BASE_DIR/src/data/measure_partition.py" \
        --graph-dirs "$TIER0_DIR" "$TIER1_DIR" \
        --max-samples "$MAX_SAMPLES"
else
    echo "ERROR: Neither TIER0_DIR nor TIER1_DIR exist." >&2
    echo "Set RUN_ROOT to a workspace containing shared_tier0_cache/shared_tier1_cache and resubmit." >&2
    exit 1
fi

echo "=========================================="
echo "Measurement finished: $(date)"
echo "=========================================="
