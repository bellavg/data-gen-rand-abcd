#!/bin/bash
#SBATCH --job-name=9_preprocess_graphs
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/preprocess_%j.out

set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
AIG_ROOT="${AIG_ROOT:-$HOME/data-gen-rand-abcd/data/designs}"
FINAL_OUT="${FINAL_OUT:-/scratch-shared/$USER/}"
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
AIG_DEBUG_PATH_COUNTS="${AIG_DEBUG_PATH_COUNTS:-0}"
FAIL_FAST="${FAIL_FAST:-1}"


# Load cluster environment directly (SLURM node).
module load 2025

source "$VENV_PATH/bin/activate"

echo "=========================================="
echo "JOB: PyG Graph Preprocessing"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Base dir: $BASE_DIR"
echo "AIG root: $AIG_ROOT"
echo "Final out: $FINAL_OUT"
echo "Venv: $VENV_PATH"
echo "Workers: $WORKERS"
echo "Debug path counts: $AIG_DEBUG_PATH_COUNTS"
echo "Fail fast: $FAIL_FAST"
echo "=========================================="

if [[ "$FAIL_FAST" == "1" ]]; then
  FAIL_FAST_FLAG="--fail-fast"
else
  FAIL_FAST_FLAG="--no-fail-fast"
fi

AIG_DEBUG_PATH_COUNTS="$AIG_DEBUG_PATH_COUNTS" \
python -m data.preprocess_data \
  --aig-root "$AIG_ROOT" \
  --final-out "$FINAL_OUT" \
  --workers "$WORKERS" \
  --allow-unmatched-names \
  "$FAIL_FAST_FLAG"

echo "Finished: $(date)"