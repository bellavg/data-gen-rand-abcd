#!/bin/bash
#SBATCH --job-name=precompute_summarization
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --array=0-31
#SBATCH --output=logs/precompute_summarization_%A_%a.out

# ---------------------------------------------------------------------------
# Precompute summarized (coarsened) graphs using Cache Manifests.
#
# The coarsened graphs are written to the node's own scratch disk and packed
# into tar.zst archives on /scratch-shared, so no per-graph inode is ever
# created on the shared filesystem — one file per graph would be ~700k inodes
# per method, which does not fit in quota.
#
# THE ARRAY INDEX CARRIES BOTH THE METHOD AND THE SHARD.
# There is deliberately no METHOD environment variable: a bare
# `METHOD=wl sbatch ...` is NOT propagated to the job on Snellius, so it
# silently ran whatever the default was.  Encoding the method in the array
# index means a plain `sbatch` is always correct.
#
#   method = METHODS[ task / SHARDS_PER_METHOD ]
#   shard  = task % SHARDS_PER_METHOD
#
# RUN EVERY METHOD (192 tasks — only once the cheap ones have been checked)
#   sbatch --array=0-191 src/shell/precompute_summarization.sh
#
# RUN ONE METHOD AT A TIME — submit only that method's index range:
#   sbatch --array=0-31    ...   # identity   (zero compression, the control)
#   sbatch --array=32-63   ...   # cone       (S1, domain-specific)
#   sbatch --array=64-95   ...   # wl         (S2, graded WL/bisimulation)
#   sbatch --array=96-127  ...   # convmatch  (S3, SOTA — the slow one)
#   sbatch --array=128-159 ...   # spectral   (S4, classic control)
#   sbatch --array=160-191 ...   # lsh        (S5, cheap control)
# The job prints its own method and range on the first lines of the log, so
# check there rather than recounting.  The default --array above runs only
# identity, because that is the cheapest end-to-end check.
#
# SMOKE TEST — one shard of one method (~1/32 of the corpus):
#   sbatch --array=0 src/shell/precompute_summarization.sh
#
# Parameters for each method come from config.SUMMARIZATION_PARAMS, not from
# this script.  The method list lives in summarization_methods.sh, shared
# with train_summarization.sh so the two can never disagree about which
# index means which method.
#
# Re-running is safe, but resume is per shard, not per graph: the working
# directory is node-local and destroyed at job end, so a shard that did not
# finish is redone from scratch.  Completion is recorded only after every
# archive is in place, so a shard interrupted mid-packing is retried rather
# than left with a missing tier.
#
# CHAIN WITH TRAINING (one method's range, then train on it)
#   PID=$(sbatch --parsable --array=32-63 src/shell/precompute_summarization.sh)
#   sbatch --dependency=afterok:$PID src/shell/train_summarization.sh
# ---------------------------------------------------------------------------

set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
source "$BASE_DIR/src/shell/summarization_methods.sh"

ALGORITHM="${ALGORITHM:-Orchestrate}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
METHOD_INDEX=$(( TASK_ID / SHARDS_PER_METHOD ))
SHARD_ID=$(( TASK_ID % SHARDS_PER_METHOD ))
NUM_SHARDS=$SHARDS_PER_METHOD
SHARD_TAG=$(printf '%03d' "$SHARD_ID")

if (( METHOD_INDEX >= ${#METHODS[@]} )); then
    echo "ERROR: array index $TASK_ID is outside every method's range." >&2
    echo "Valid ranges are:" >&2
    summarization_ranges | while read -r name first last; do
        printf '  --array=%s-%s  %s\n' "$first" "$last" "$name" >&2
    done
    exit 1
fi

METHOD="${METHODS[$METHOD_INDEX]}"

echo "=========================================="
echo "PRECOMPUTE SUMMARIZATION JOB: ${SLURM_ARRAY_JOB_ID:-local} task ${TASK_ID}"
echo "Method: $METHOD  (this method is --array=$(( METHOD_INDEX * SHARDS_PER_METHOD ))-$(( (METHOD_INDEX + 1) * SHARDS_PER_METHOD - 1 )))"
echo "Shard: ${SHARD_ID}/${NUM_SHARDS}"
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

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. PATHS
# =========================================================

MANIFEST_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache/metadata"
ARCHIVE_DIR="/scratch-shared/$USER/aig_summary_cache/${METHOD}"
WORK_DIR="$TMPDIR/summary_${METHOD}"

mkdir -p "$ARCHIVE_DIR" "$WORK_DIR"

# Gate on a sentinel written only after every archive is in place.  Checking
# for the archives themselves would skip a shard that died between packing
# tier0 and tier1, leaving that tier missing forever.
DONE_SENTINEL="$ARCHIVE_DIR/.shard${SHARD_TAG}.done"
if [[ -f "$DONE_SENTINEL" ]]; then
    echo "[shard ${SHARD_ID}] Already complete — skipping."
    exit 0
fi

# A previous attempt may have left some archives behind; they are rewritten
# below, but drop them first so a crash cannot leave a mixed-vintage set.
rm -f "$ARCHIVE_DIR"/*_shard"${SHARD_TAG}".tar.zst \
      "$ARCHIVE_DIR"/*_shard"${SHARD_TAG}".tar.zst.part

echo "Manifests:  $MANIFEST_DIR"
echo "Work dir:   $WORK_DIR  (node-local)"
echo "Archives:   $ARCHIVE_DIR"

# =========================================================
# 3. SUMMARIZE INTO NODE-LOCAL SCRATCH
# =========================================================

time python -W ignore -u -m data.summarize_graphs "$METHOD" \
    --manifest-dirs "$MANIFEST_DIR" \
    --out-dir "$WORK_DIR" \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS"

# =========================================================
# 4. PACK ONE ARCHIVE PER CACHE DIRECTORY
# =========================================================
# Written to a .part name first so an interrupted job never leaves an archive
# that the skip-check above would mistake for completed work.

packed=0
for sub in "$WORK_DIR"/*/; do
    [[ -d "$sub" ]] || continue
    name=$(basename "$sub")
    archive="$ARCHIVE_DIR/${name}_shard${SHARD_TAG}.tar.zst"
    echo "Packing $name -> $(basename "$archive")"
    tar --zstd -cf "${archive}.part" -C "$sub" .
    mv "${archive}.part" "$archive"
    packed=$((packed + 1))
done

if [[ "$packed" -eq 0 ]]; then
    echo "ERROR: no cache directories were produced in $WORK_DIR" >&2
    exit 1
fi

# Per-shard stats stay outside the archives so they can be read without
# unpacking (compression ratios and wall-clock feed the RQ2 table).
cp "$WORK_DIR"/_summary_stats_*.json "$ARCHIVE_DIR/"

touch "$DONE_SENTINEL"

echo "=========================================="
echo "Shard ${SHARD_ID} complete: $packed archive(s)."
ls -lh "$ARCHIVE_DIR"/*_shard"${SHARD_TAG}".tar.zst
echo "End time: $(date)"
echo "=========================================="
