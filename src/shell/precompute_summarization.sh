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
# Each array task handles one shard.  The coarsened graphs are written to the
# node's own scratch disk and packed into tar.zst archives on /scratch-shared,
# so no per-graph inode is ever created on the shared filesystem — one file
# per graph would be ~700k inodes per method, which does not fit in quota.
#
#   METHOD=identity sbatch src/shell/precompute_summarization.sh
#
# METHOD is one of identity / cone / wl / convmatch / spectral / lsh; the
# parameters each one runs with come from config.SUMMARIZATION_PARAMS, not
# from this script.  One run per method, then train_summarization.sh.
#
# Re-running is safe, but resume is per shard, not per graph: the working
# directory is node-local and destroyed at job end, so a shard that did not
# finish is redone from scratch.  Completion is recorded only after every
# archive is in place, so a shard interrupted mid-packing is retried rather
# than left with a missing tier.
#
# To repair one shard, override NUM_SHARDS to match the original array size:
#   NUM_SHARDS=32 sbatch --array=7 src/shell/precompute_summarization.sh
#
# CHAIN WITH TRAINING
#   PID=$(sbatch --parsable src/shell/precompute_summarization.sh)
#   sbatch --dependency=afterok:$PID src/shell/train_summarization.sh
# ---------------------------------------------------------------------------

set -euo pipefail

METHOD="${METHOD:-identity}"
ALGORITHM="${ALGORITHM:-Orchestrate}"

SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
# Slurm reports SLURM_ARRAY_TASK_COUNT=1 when a single index is resubmitted
# (--array=7), which would make the shard slice wrong, hence the override.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_TAG=$(printf '%03d' "$SHARD_ID")

echo "=========================================="
echo "PRECOMPUTE SUMMARIZATION JOB: ${SLURM_ARRAY_JOB_ID:-local} shard ${SHARD_ID}/${NUM_SHARDS}"
echo "Method: $METHOD"
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
