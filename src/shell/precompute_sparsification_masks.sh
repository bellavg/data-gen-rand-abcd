#!/bin/bash
#SBATCH --job-name=precompute_sparsification_masks
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/precompute_sparsification_%j.out

# ---------------------------------------------------------------------------
# Precompute sparsification masks using Cache Manifests
# ---------------------------------------------------------------------------

set -euo pipefail

# Define the algorithm argument (override via env var to run this once per
# method, e.g. SPARSIFICATION_ALGO=pagerank sbatch precompute_sparsification_masks.sh)
SPARSIFICATION_ALGO="${SPARSIFICATION_ALGO:-and_gate_only}"


# Match the workspace targeted in train.sh
ALGORITHM="${ALGORITHM:-Orchestrate}"

# Workspace root. Defaults to the EVAL workspace (see EVALUATION.md) — masks
# for training were precomputed long ago, so eval is the live use case and the
# safe default. To (re)build masks for TRAINING you must pass the train root
# explicitly: RUN_ROOT=/scratch-shared/$USER/aig_train_run
RUN_ROOT="${RUN_ROOT:-/scratch-shared/$USER/aig_eval_run}"
# Strip a trailing slash so the train-root comparison below is exact-match
# safe: a hand-typed ".../aig_train_run/" would otherwise miss the redirect
# branch and write masks in-place, where the config lookup won't find them.
RUN_ROOT="${RUN_ROOT%/}"

# Mask redirect: training writes sparsification masks OFF the main cache (to a
# separate aig_mask_cache) to cut inode pressure, and the dataset looks them up
# there via config.SPARSIFICATION_REPLACE_PATH (which only rewrites the train
# root). For any other RUN_ROOT that config redirect is a no-op, so masks must
# be written IN-PLACE or the lookup won't find them — hence the branch below
# redirects only when RUN_ROOT is the train root, whichever way it was set.
if [[ "$RUN_ROOT" == "/scratch-shared/$USER/aig_train_run" ]]; then
    MASK_CACHE_ROOT="${MASK_CACHE_ROOT:-/scratch-shared/$USER/aig_mask_cache}"
else
    MASK_CACHE_ROOT="${MASK_CACHE_ROOT:-$RUN_ROOT}"
fi

echo "=========================================="
echo "PRECOMPUTE SPARSIFICATION MASKS JOB ID: $SLURM_JOB_ID"
echo "Algorithm: $ALGORITHM"
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

# BASE_DIR matches the repo root, just like in training
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. MANIFEST DIRECTORIES
# =========================================================

MANIFEST_DIR="$RUN_ROOT/${ALGORITHM}/cache/metadata"

echo "=========================================="
echo "Processing manifests in $MANIFEST_DIR"
echo "Mask output root: $MASK_CACHE_ROOT (in-place if == RUN_ROOT)"
echo "=========================================="

# Redirect masks only when MASK_CACHE_ROOT differs from RUN_ROOT (training);
# for the eval workspace they are written in-place (no --replace-path).
if [[ "$MASK_CACHE_ROOT" == "$RUN_ROOT" ]]; then
    REPLACE_ARGS=()
else
    REPLACE_ARGS=(--replace-path "$RUN_ROOT" "$MASK_CACHE_ROOT")
fi

# Call the updated python script with --manifest-dirs
time python -W ignore -u -m data.sparsification "$SPARSIFICATION_ALGO" \
    --manifest-dirs "$MANIFEST_DIR" \
    ${REPLACE_ARGS[@]+"${REPLACE_ARGS[@]}"}

echo "=========================================="
echo "Precomputation complete."
echo "End time: $(date)"
echo "=========================================="
