#!/bin/bash
#SBATCH --job-name=aig_train_summarization
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --array=0-0
#SBATCH --output=logs/train_summarization_%A_%a.out
#
# Train on summarized (coarsened) graphs.
#
# The graphs were materialized offline by precompute_summarization.sh, so this
# job does no coarsening: it unpacks the archives to the node's own disk and
# then runs the ordinary training path.  Unpacking costs a few minutes against
# a 48-hour job, whereas rewriting graphs inside the dataloader would cost CPU
# on the GPU node every epoch.
#
#   PID=$(sbatch --parsable src/shell/precompute_summarization.sh)
#   sbatch --dependency=afterok:$PID src/shell/train_summarization.sh
#
# The staging directory deliberately contains no job id.  dataset.py hashes the
# tier cache paths into its cache signature and stores absolute cache paths in
# the manifest, so a per-job path would change the signature every run and
# force a full cache rebuild on the GPU node.

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Methods to train, one per array task.  Extend as methods are implemented.
METHODS=("identity")
METHOD=${METHODS[${SLURM_ARRAY_TASK_ID:-0}]}
ALGORITHM="Orchestrate"

echo "=========================================="
echo "JOB ARRAY ID: ${SLURM_ARRAY_JOB_ID:-local}, TASK ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Summarization method: $METHOD"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# =========================================================
# 1. Setup Environment & Modules
# =========================================================

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# TCMalloc prevents glibc malloc fragmentation over long training runs.
module load gperftools/2.16-GCCcore-14.2.0
export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export AIG_REQUIRE_GPU=1
export WANDB_INIT_TIMEOUT=120

# Prevent worker-thread thrashing in data-loader subprocesses.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# =========================================================
# 2. STAGE THE SUMMARIZED GRAPHS TO NODE-LOCAL DISK
# =========================================================

ARCHIVE_DIR="/scratch-shared/$USER/aig_summary_cache/${METHOD}"
STAGE_DIR="/scratch-node/$USER/aig_summary/${METHOD}"
UNTAR_JOBS="${UNTAR_JOBS:-16}"

# Existence of the directory proves nothing — precompute_summarization.sh
# creates it before doing any work.  Require actual archives, because a
# missing tier makes dataset.py silently fall back to caching the raw,
# UNSUMMARIZED graph into the empty staging directory.
if ! compgen -G "$ARCHIVE_DIR/*_shard*.tar.zst" > /dev/null; then
    echo "ERROR: no summarized archives in $ARCHIVE_DIR" >&2
    echo "Run: METHOD=$METHOD sbatch src/shell/precompute_summarization.sh" >&2
    exit 1
fi

mkdir -p "$STAGE_DIR"

# Concurrent array tasks can land on the same node, so serialize the unpack
# and let the sentinel make it a no-op for everyone after the first.
(
    flock 9
    if [[ -f "$STAGE_DIR/.ready" ]]; then
        echo "[stage] Already staged at $STAGE_DIR"
    else
        echo "[stage] Unpacking $ARCHIVE_DIR -> $STAGE_DIR"
        stage_start=$SECONDS
        for archive in "$ARCHIVE_DIR"/*_shard*.tar.zst; do
            name=$(basename "$archive")
            name=${name%_shard*}
            mkdir -p "$STAGE_DIR/$name"
            echo "$archive" "$STAGE_DIR/$name"
        done | xargs -P "$UNTAR_JOBS" -n 2 sh -c 'tar --zstd -xf "$0" -C "$1"'

        # Each shard wrote its own node-count index; the dataset reads a single
        # merged _num_nodes_global.json per cache directory.
        for sub in "$STAGE_DIR"/*/; do
            [[ -d "$sub" ]] || continue
            python -c "
import sys
from data.summarize_graphs import merge_shard_indexes
n = merge_shard_indexes(sys.argv[1])
print('[stage] merged', n, 'entries in', sys.argv[1])
sys.exit(0 if n else 1)
" "$sub"
        done

        touch "$STAGE_DIR/.ready"
        echo "[stage] Done in $((SECONDS - stage_start))s"
    fi
) 9> "$STAGE_DIR.lock"

# Both tiers must be present and non-empty.  Without this an incomplete
# precompute trains on raw graphs re-cached by dataset.py, producing a run
# that looks healthy and measures nothing.
for required in shared_tier0_cache shared_tier1_cache; do
    index="$STAGE_DIR/$required/_num_nodes_global.json"
    if [[ ! -s "$index" ]]; then
        echo "ERROR: staging incomplete — missing or empty $index" >&2
        echo "Delete $STAGE_DIR and re-run precompute for method '$METHOD'." >&2
        exit 1
    fi
done

du -sh "$STAGE_DIR"

# =========================================================
# 3. DATA & OUTPUT PATHS
# =========================================================

CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

# Checkpoints and logs stay on shared scratch; only the graph cache is local.
WORKSPACE="/scratch-shared/$USER/aig_summary_run/${ALGORITHM}/${METHOD}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# cache_dir covers both the metadata (manifest/splits/batch plans) and the
# non-tier graphs under processed_graphs/, and those graphs are the summarized
# ones, so the whole directory has to be the staging dir.  The metadata is
# cheap to rebuild once per node; the alternative silently trains on
# unsummarized tier-2 graphs.
CACHE_DIR="$STAGE_DIR"
TIER0_CACHE_DIR="$STAGE_DIR/shared_tier0_cache"
TIER1_CACHE_DIR="$STAGE_DIR/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
# All 50K graphs used across both HP tuning stages (15K Stage-1 + 35K Stage-2).
# Using this file ensures zero HP tuning leakage into final train/val/test splits.
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

if [ ! -f "$HP_TUNING_SPLITS" ]; then
    echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"
fi

# =========================================================
# 4. Runtime settings
# =========================================================

NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-true}"

echo "Using NUM_WORKERS=$NUM_WORKERS, PREFETCH_FACTOR=$PREFETCH_FACTOR."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

# =========================================================
# 5. EXECUTE TRAINING
# =========================================================

echo "Starting training for $ALGORITHM on summarization=$METHOD ..."

srun python -u -m train \
    --algorithm         "$ALGORITHM" \
    --csv_paths         "$CSV_PATH" \
    --checkpoint_dir    "$CHECKPOINT_DIR" \
    --log_dir           "$LOG_DIR" \
    --cache_dir         "$CACHE_DIR" \
    --tier0_cache_dir   "$TIER0_CACHE_DIR" \
    --tier1_cache_dir   "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --prefetch_factor   "$PREFETCH_FACTOR" \
    --num_workers       "$NUM_WORKERS" \
    --pin_memory        "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --torch_compile     "$TORCH_COMPILE" \
    --patience          4

echo "=========================================="
echo "Training for $ALGORITHM ($METHOD) complete."
echo "End time: $(date)"
echo "=========================================="
