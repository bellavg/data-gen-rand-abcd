#!/bin/bash
#SBATCH --job-name=big_optuna
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=1-3
#SBATCH --output=logs/big_optuna_worker_%a_s2.out
# NOTE: SBATCH directives cannot reference shell variables, so override --output
# at submit time to avoid overwriting previous stage logs:
#   Stage 1: sbatch --output="logs/big_optuna_worker_%a_s1.out" src/shell/big_hp_tuning.sh
#   Stage 2: STAGE=2 sbatch --output="logs/big_optuna_worker_%a_s2.out" src/shell/big_hp_tuning.sh

set -euo pipefail

# Array-task id, or 1 when run outside an array for local debugging.
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

SCRIPT_VERSION="2026-05-14 (72h Two-Stage Array Job: Stage 1 = explore, Stage 2 = seed+exploit)"

echo "=========================================="
echo "JOB: Big Optuna Hyperparameter Tuning (array_task=${TASK_ID} job=${SLURM_ARRAY_JOB_ID:-n/a})"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Script version: $SCRIPT_VERSION"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# --- Load gperftools TCMalloc to prevent glibc malloc fragmentation.
# TCMalloc aggressively returns freed memory to the OS so that thousands of
# sequential torch.load() calls over a long tuning run do not silently balloon
# the process RSS until SLURM kills it.
module load gperftools/2.16-GCCcore-14.2.0
export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "TCMalloc preloaded from: ${EBROOTGPERFTOOLS}/lib/libtcmalloc.so"
# -------------------------------------------------------------------------

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Skip pip installations since the venv is already set up!

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$BASE_DIR"

# 4. Define Output Paths in Scratch
# Each array task has its own workspace and study to avoid DB locking and
# log-file conflicts. All tasks share one preprocessed dataset cache directory
# so graph preprocessing is done at most once across the three workers.
#
# Two-stage tuning strategy (set STAGE=1 or STAGE=2 at submission time):
#   Stage 1  — fast exploration: 15K samples × 50 trials per worker
#              1 h/trial cap → ~50 h worst-case; finds promising HP regions
#   Stage 2  — seeded exploitation: 35K samples × 20 trials per worker
#              2 h/trial cap → ~40 h worst-case; top-SEED_TOP_N Stage-1 configs
#              enqueued first, remainder are new TPE-guided trials
# 72h job limit → both stages comfortably fit. In practice (patience=3,
# BF16 pruning): Stage 1 ~25-30h, Stage 2 ~25-35h per worker.
STAGE="${STAGE:-1}"
if [[ "$STAGE" == "1" ]]; then
    TRAIN_SAMPLES=15000
    N_TRIALS=50
    MAX_TRIAL_HOURS=1.0
elif [[ "$STAGE" == "2" ]]; then
    TRAIN_SAMPLES=35000
    N_TRIALS=20
    MAX_TRIAL_HOURS=2.0
else
    echo "ERROR: STAGE must be 1 or 2 (got '$STAGE')." >&2
    exit 1
fi
echo "Stage: $STAGE  train_samples=$TRAIN_SAMPLES  n_trials=$N_TRIALS  max_trial_hours=$MAX_TRIAL_HOURS"

WORKSPACE="/scratch-shared/$USER/big_optuna_run_s${STAGE}_${TASK_ID}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"

# Common cache shared by all array workers (read-only after first warm-up).
SHARED_CACHE="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache"

# Study name is worker-scoped but stage-consistent so Stage-2 warm-start works.
STUDY_NAME="${STUDY_NAME:-big_optuna_hp_tuning}_worker${TASK_ID}"

# Use node-local scratch for SQLite to avoid GPFS file-lock hangs.
# The DB is copied to WORKSPACE at the end so results are preserved.
LOCAL_SCRATCH="${TMPDIR:-/tmp}/optuna_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_SCRATCH"
STORAGE_PATH="sqlite:///$LOCAL_SCRATCH/optuna_study.db"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$SHARED_CACHE"

# 5. Define Input CSVs
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# ---------------------------------------------------------
# CACHE SENTINEL CHECK
# The dedicated warmup_cache.sh job should be submitted and completed before
# this array job.  All 3 tasks simply wait until the sentinel is present;
# they never build the cache themselves (avoids GPFS contention and wasted
# GPU-node time on file I/O).
# If warmup_cache.sh was not run first, fail fast after timeout instead of
# letting all GPU workers start cold and stampede shared storage.
# ---------------------------------------------------------
CACHE_READY_SENTINEL="$SHARED_CACHE/cache_ready_n${TRAIN_SAMPLES}.sentinel"

if [[ -f "$CACHE_READY_SENTINEL" ]]; then
    echo "Cache warm (sentinel found). Task $TASK_ID proceeding immediately."
else
    echo "Task $TASK_ID: sentinel not found — waiting up to 30 min for cache..."
    echo "  (Run warmup_cache.sh before big_hp_tuning.sh to avoid this wait.)"
    MAX_WAIT=1800
    WAITED=0
    while [[ ! -f "$CACHE_READY_SENTINEL" ]]; do
        sleep 30
        WAITED=$((WAITED + 30))
        if [[ $((WAITED % 300)) -eq 0 ]]; then
            echo "  ... still waiting (${WAITED}s elapsed)"
        fi
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "ERROR: timed out waiting for cache sentinel '$CACHE_READY_SENTINEL'." >&2
            echo "Submit warmup_cache.sh first, or chain via --dependency=afterok." >&2
            exit 1
        fi
    done
    echo "Task $TASK_ID proceeding."
fi
# ---------------------------------------------------------

# Stage 2: seed a fresh study with the top-SEED_TOP_N completed Stage-1 configs.
# Those are enqueued first so Stage 2 immediately re-evaluates the most promising
# regions on the larger dataset; TPE then explores adjacent space for the
# remaining (N_TRIALS - SEED_TOP_N) trials.
SEED_TOP_N="${SEED_TOP_N:-10}"
SEED_FLAGS=()
if [[ "$STAGE" == "2" ]]; then
    S1_DB="/scratch-shared/$USER/big_optuna_run_s1_${TASK_ID}/optuna_study.db"
    if [[ -f "$S1_DB" ]]; then
        echo "Stage 2: seeding from top-${SEED_TOP_N} Stage-1 trials (DB: $S1_DB)"
        SEED_FLAGS=(
            --seed_from_db_url "sqlite:///$S1_DB"
            --seed_study_name "$STUDY_NAME"
            --seed_top_n "$SEED_TOP_N"
        )
    else
        echo "WARNING: Stage-1 DB not found at $S1_DB; starting Stage 2 cold." >&2
    fi
fi
echo "Seed flags: ${SEED_FLAGS[*]:-(none)}"
echo "Seed top-N: $SEED_TOP_N"

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER (big run)
# ---------------------------------------------------------
# Each worker uses a distinct sampler seed so TPE explores a different
# region of the HP space. dataset_seed stays fixed so the train/val split
# is identical across all workers (comparable evaluation).
SAMPLER_SEED=$((40 + TASK_ID))   # workers 1/2/3 → seeds 41/42/43

# Default to small parallelism to reduce main-process fragmentation.
NUM_WORKERS="${NUM_WORKERS:-2}"

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PREFETCH_FACTOR=1
DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
MEMORY_TELEMETRY_TRIALS="${MEMORY_TELEMETRY_TRIALS:-0}"

if [ "$NUM_WORKERS" -eq 0 ]; then
    PERSISTENT_WORKERS=false
fi

EXTRA_FLAGS=()
if [ "$PIN_MEMORY" = "true" ]; then
    EXTRA_FLAGS+=(--pin_memory)
fi
if [ "$PERSISTENT_WORKERS" = "true" ]; then
    EXTRA_FLAGS+=(--persistent_workers)
fi
if [ "$NUM_WORKERS" -gt 0 ]; then
    EXTRA_FLAGS+=(--prefetch_factor "$PREFETCH_FACTOR")
fi
if [ "$DYNAMIC_BATCHING" = "true" ]; then
    EXTRA_FLAGS+=(--dynamic_batching)
fi

echo "DataLoader config: num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY persistent_workers=$PERSISTENT_WORKERS prefetch_factor=$PREFETCH_FACTOR dynamic_batching=$DYNAMIC_BATCHING"
echo "Memory telemetry: first $MEMORY_TELEMETRY_TRIALS trial(s)"

echo "Starting Big Worker $TASK_ID on GPU 0 (stage=$STAGE sampler_seed=$SAMPLER_SEED, study=$STUDY_NAME)..."

sync_optuna_db() {
    local src_db="$1"
    local dst_tmp="$2"

    if [ ! -f "$src_db" ]; then
        return 2  # nothing to sync yet; not an error
    fi

    python - "$src_db" "$dst_tmp" <<'PYEOF'
import sqlite3
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)

src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
dst_conn = sqlite3.connect(str(dst))
try:
    with dst_conn:
        src_conn.backup(dst_conn)
finally:
    dst_conn.close()
    src_conn.close()
PYEOF
}

SYNC_PID=""
cleanup_sync_loop() {
    if [ -n "${SYNC_PID:-}" ]; then
        kill "$SYNC_PID" 2>/dev/null || true
        wait "$SYNC_PID" 2>/dev/null || true
    fi
}
trap cleanup_sync_loop EXIT INT TERM

# Background sync loop: copy the node-local SQLite DB to scratch-shared every
# 5 minutes so that completed trials are preserved even if the job crashes.
DB_SYNC_INTERVAL="${DB_SYNC_INTERVAL:-3600}"
(
    while true; do
        sleep "$DB_SYNC_INTERVAL"
        _sync_rc=0
        sync_optuna_db "$LOCAL_SCRATCH/optuna_study.db" "$WORKSPACE/optuna_study.db.tmp" || _sync_rc=$?
        if [ "$_sync_rc" -eq 0 ]; then
            if mv "$WORKSPACE/optuna_study.db.tmp" "$WORKSPACE/optuna_study.db"; then
                echo "[db_sync] $(date) synced DB to $WORKSPACE/optuna_study.db"
            else
                echo "[db_sync] $(date) WARNING: atomic rename failed" >&2
            fi
        elif [ "$_sync_rc" -ne 2 ]; then
            echo "[db_sync] $(date) WARNING: SQLite backup failed" >&2
        fi
    done
) &
SYNC_PID=$!

EXIT_CODE=0
python -u -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "$STUDY_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$SHARED_CACHE" \
    --log_dir "$LOG_DIR/worker_${TASK_ID}" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    ${SEED_FLAGS[@]+"${SEED_FLAGS[@]}"} \
    --memory_guard_max_tokens 3.5e8 \
    --dataset_seed 42 \
    --sampler_seed "$SAMPLER_SEED" \
    --memory_telemetry_trials "$MEMORY_TELEMETRY_TRIALS" \
    --train_samples "$TRAIN_SAMPLES" \
    --max_trial_hours "$MAX_TRIAL_HOURS" \
    --n_trials "$N_TRIALS" \
    > "$LOG_DIR/worker_${TASK_ID}.log" 2>&1 || EXIT_CODE=$?

# Stop the background sync loop
cleanup_sync_loop
SYNC_PID=""

# Final copy regardless of exit code
echo "Final sync of SQLite DB to $WORKSPACE/optuna_study.db ..."
_final_rc=0
sync_optuna_db "$LOCAL_SCRATCH/optuna_study.db" "$WORKSPACE/optuna_study.db.tmp" || _final_rc=$?
if [ "$_final_rc" -eq 0 ]; then
    mv "$WORKSPACE/optuna_study.db.tmp" "$WORKSPACE/optuna_study.db" \
        || echo "WARNING: final DB rename failed" >&2
elif [ "$_final_rc" -ne 2 ]; then
    echo "WARNING: final DB sync failed" >&2
fi

if (( EXIT_CODE != 0 )); then
    echo "Worker $TASK_ID failed (exit $EXIT_CODE). Last 80 lines:"
    tail -n 80 "$LOG_DIR/worker_${TASK_ID}.log" || true
    exit "$EXIT_CODE"
fi

echo "=========================================="
echo "BIG JOB complete (worker $TASK_ID)."
echo "End time: $(date)"
