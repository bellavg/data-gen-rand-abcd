#!/bin/bash
#SBATCH --job-name=9_preprocess_graphs
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --constraint=scratch-node
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
STAGE_TO_SCRATCH="${STAGE_TO_SCRATCH:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10000}"


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
echo "Stage to scratch: $STAGE_TO_SCRATCH"
echo "Progress every: $PROGRESS_EVERY"
echo "=========================================="

if [[ -n "${TMPDIR:-}" ]]; then
  LOCAL_SCRATCH="$TMPDIR"
else
  LOCAL_SCRATCH="/scratch-shared/$USER/tmp"
fi
mkdir -p "$LOCAL_SCRATCH"

WORK_DIR=$(mktemp -d "$LOCAL_SCRATCH/preprocess_aigs_${SLURM_JOB_ID:-manual}_XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

STAGED_AIG_ROOT="$WORK_DIR/designs"

if [[ "$STAGE_TO_SCRATCH" == "1" ]]; then
  echo "Staging tier0/tier1 AIG zips into local scratch: $STAGED_AIG_ROOT"
  mkdir -p "$STAGED_AIG_ROOT"

  for design_dir in "$AIG_ROOT"/*; do
    [[ -d "$design_dir" ]] || continue
    design_name=$(basename "$design_dir")

    # Skip non-design metadata folders if present at root.
    if [[ "$design_name" == "design_metadata" || "$design_name" == "logs" ]]; then
      continue
    fi

    out_tier0="$STAGED_AIG_ROOT/$design_name/tier0"
    out_tier1="$STAGED_AIG_ROOT/$design_name/tier1"
    mkdir -p "$out_tier0" "$out_tier1"

    # Tier0 archive locations used in this project.
    if [[ -f "$design_dir/tier0.zip" ]]; then
      unzip -q -o "$design_dir/tier0.zip" -d "$out_tier0"
    elif [[ -f "$design_dir/tier0/tier0.zip" ]]; then
      unzip -q -o "$design_dir/tier0/tier0.zip" -d "$out_tier0"
    fi

    # Tier1 data archives: one zip per algorithm.
    if [[ -d "$design_dir/tier1" ]]; then
      for t1_zip in "$design_dir"/tier1/*.zip; do
        [[ -f "$t1_zip" ]] || continue
        unzip -q -o "$t1_zip" -d "$out_tier1"
      done
    fi
  done

  # Flatten accidental nested paths from archives while keeping tier dirs.
  find "$STAGED_AIG_ROOT" -type f -name "*.aig" | while read -r aig_file; do
    parent_dir=$(basename "$(dirname "$aig_file")")
    if [[ "$parent_dir" == "tier0" || "$parent_dir" == "tier1" ]]; then
      continue
    fi
    rel_path=${aig_file#"$STAGED_AIG_ROOT/"}
    design_name=${rel_path%%/*}
    if [[ "$aig_file" == *"_tier1_"* ]]; then
      target_dir="$STAGED_AIG_ROOT/$design_name/tier1"
    else
      target_dir="$STAGED_AIG_ROOT/$design_name/tier0"
    fi
    mkdir -p "$target_dir"
    mv -f "$aig_file" "$target_dir/$(basename "$aig_file")"
  done

  STAGED_COUNT=$(find "$STAGED_AIG_ROOT" -type f -name "*.aig" | wc -l)
  echo "Staging complete: staged_aigs=$STAGED_COUNT"
  EFFECTIVE_AIG_ROOT="$STAGED_AIG_ROOT"
else
  EFFECTIVE_AIG_ROOT="$AIG_ROOT"
fi

if [[ "$FAIL_FAST" == "1" ]]; then
  FAIL_FAST_FLAG="--fail-fast"
else
  FAIL_FAST_FLAG="--no-fail-fast"
fi

AIG_DEBUG_PATH_COUNTS="$AIG_DEBUG_PATH_COUNTS" \
python -u -m data.preprocess_data \
  --aig-root "$EFFECTIVE_AIG_ROOT" \
  --final-out "$FINAL_OUT" \
  --workers "$WORKERS" \
  --progress-every "$PROGRESS_EVERY" \
  --allow-unmatched-names \
  "$FAIL_FAST_FLAG"

echo "Finished: $(date)"