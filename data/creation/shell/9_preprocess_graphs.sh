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
echo "=========================================="

python -m data.preprocess_data \
  --aig-root "$AIG_ROOT" \
  --final-out "$FINAL_OUT" \
  --workers "$WORKERS" \
  --allow-unmatched-names \
  --progress

echo "Finished: $(date)"