#!/bin/bash
#SBATCH --job-name=refresh_scratch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=genoa
#SBATCH --output=logs/refresh_scratch_%j.out

# ---------------------------------------------------------------------------
# Refresh all files in /scratch-shared/<user>/ so they are not purged by the
# 14-day automatic deletion policy (files older than 14 days are removed).
#
# Strategy: touch every file under the scratch directories to update mtime.
# This is far faster than copying and uses zero extra disk quota.
# Run this via SLURM cron / --dependency or submit manually every ~10 days.
#
# Usage:
#   sbatch src/shell/refresh_scratch.sh
#
# Override defaults with environment variables:
#   SCRATCH_ROOT=/scratch-shared/$USER sbatch src/shell/refresh_scratch.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/$USER}"

echo "=========================================="
echo "SCRATCH REFRESH JOB"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Scratch root: $SCRATCH_ROOT"
echo "=========================================="

if [[ ! -d "$SCRATCH_ROOT" ]]; then
    echo "ERROR: scratch root does not exist: $SCRATCH_ROOT" >&2
    exit 1
fi

refresh_dir() {
    local dir="$1"
    local label="$2"
    if [[ ! -d "$dir" ]]; then
        echo "[refresh] SKIP (not found): $dir"
        return
    fi
    local count
    count=$(find "$dir" -type f | wc -l)
    echo "[refresh] touching $count files in $dir ($label) ..."
    # Use find + touch in batches via xargs for speed.
    find "$dir" -type f -print0 | xargs -0 -P 4 touch --
    echo "[refresh] done: $dir"
}

# Preprocessed PyG graphs (the most critical — these take hours to rebuild).
refresh_dir "$SCRATCH_ROOT/graphs" "PyG .pt graphs"

# Dataset cache (split JSONs, node-size cache, warmup sentinel files).
refresh_dir "$SCRATCH_ROOT/big_optuna_run/shared_dataset_cache" "warmup dataset cache"

# Optuna run workspaces (stage 1 + 2 DBs for all workers).
for stage_dir in "$SCRATCH_ROOT"/big_optuna_run_s*; do
    [[ -d "$stage_dir" ]] && refresh_dir "$stage_dir" "optuna workspace $(basename "$stage_dir")"
done

# Virtual environment.
refresh_dir "$SCRATCH_ROOT/.venv" "python venv"

echo "=========================================="
echo "SCRATCH REFRESH complete."
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "NOTE: submit this script again before the 14-day purge window."
echo "Recommended: run every 10 days (cron or --dependency=afterok chain)."
echo ""
echo "To schedule the next refresh automatically:"
echo "  sbatch --begin=now+10days src/shell/refresh_scratch.sh"
