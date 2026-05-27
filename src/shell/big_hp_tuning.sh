#!/bin/bash
#SBATCH --job-name=big_optuna
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=1-3
#SBATCH --output=logs/big_optuna_worker_%A_%a_s2.out
# NOTE: SBATCH directives cannot reference shell variables, so override --output
# at submit time to avoid overwriting previous stage logs.  %A = array master job
# ID, %a = array task ID — together they make each submission's .out unique.
#   Stage 1: sbatch --output="logs/big_optuna_worker_%A_%a_s1.out" src/shell/big_hp_tuning.sh
#   Stage 2: STAGE=2 sbatch --output="logs/big_optuna_worker_%A_%a_s2.out" src/shell/big_hp_tuning.sh

set -euo pipefail

# Array-task id, or 1 when run outside an array for local debugging.
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

SCRIPT_VERSION="2026-05-24 (48h Two-Stage Array Job: Stage 1 = explore, Stage 2 = seed+exploit)"

echo "=========================================="
echo "JOB: Big Optuna Hyperparameter Tuning (array_task=${TASK_ID} job=${SLURM_ARRAY_JOB_ID:-n/a})"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "SLURM memory: mem_per_node=${SLURM_MEM_PER_NODE:-unset} mem_per_cpu=${SLURM_MEM_PER_CPU:-unset}"
echo "Script version: $SCRIPT_VERSION"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

# --- Allocator selection (override via USE_TCMALLOC=false to use glibc).
# smaps analysis showed TCMalloc holds pages as "in-use" across steps and
# MallocExtension_ReleaseFreeMemory does NOT return them (private_dirty stays
# constant at 6.5+ GiB even after release calls).  Testing without TCMalloc
# lets glibc's allocator handle the workload, where malloc_trim(0) IS
# effective at returning free pages to the OS.
USE_TCMALLOC="${USE_TCMALLOC:-true}"
if [[ "$USE_TCMALLOC" == "true" ]]; then
    module load gperftools/2.16-GCCcore-14.2.0
    export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"
    export TCMALLOC_RELEASE_RATE=1000
    echo "TCMalloc preloaded from: ${EBROOTGPERFTOOLS}/lib/libtcmalloc.so"
else
    echo "TCMalloc disabled (USE_TCMALLOC=false); using glibc allocator + malloc_trim"
fi
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

# 2. Stage Config & Output Paths
# Each array task has its own workspace and study to avoid DB locking and
# log-file conflicts. All tasks share one preprocessed dataset cache directory
# so graph preprocessing is done at most once across the three workers.
#
# Two-stage tuning strategy (set STAGE=1 or STAGE=2 at submission time):
#   Stage 1  — fast exploration: 15K samples × 50 trials per worker
#              1 h/trial cap → ~50 h worst-case; finds promising HP regions
#   Stage 2  — seeded exploitation: 35K samples × 20 trials per worker
#              4 h/trial cap → ~80 h worst-case; top-SEED_TOP_N Stage-1 configs
#              enqueued first, remainder are new TPE-guided trials
# 48h job limit — each stage fits in one submission.
STAGE="${STAGE:-2}"
if [[ "$STAGE" == "1" ]]; then
    TRAIN_SAMPLES=15000
    N_TRIALS=50
    MAX_TRIAL_HOURS=1.0
elif [[ "$STAGE" == "2" ]]; then
    TRAIN_SAMPLES=35000
    N_TRIALS=20
    MAX_TRIAL_HOURS=4.0
else
    echo "ERROR: STAGE must be 1 or 2 (got '$STAGE')." >&2
    exit 1
fi

# Stage-specific memory defaults. Stage 2 keeps safety guards enabled but uses
# larger H100-oriented limits and less restrictive dynamic buckets.
if [[ "$STAGE" == "2" ]]; then
    MEMORY_GUARD_MAX_TOKENS="${MEMORY_GUARD_MAX_TOKENS:-1.2e10}"
    HARD_PRUNE_RISK="${HARD_PRUNE_RISK:-1.5e10}"
    DYNAMIC_BUCKET_RULES="${DYNAMIC_BUCKET_RULES:-240000:1,160000:1,100000:2}"
else
    MEMORY_GUARD_MAX_TOKENS="${MEMORY_GUARD_MAX_TOKENS:-2.5e8}"
    HARD_PRUNE_RISK="${HARD_PRUNE_RISK:-1e10}"
    DYNAMIC_BUCKET_RULES="${DYNAMIC_BUCKET_RULES:-}"
fi
HARD_PRUNE="${HARD_PRUNE:-true}"
MEMORY_RELEASE_INTERVAL_STEPS="${MEMORY_RELEASE_INTERVAL_STEPS:-200}"
echo "Stage: $STAGE  train_samples=$TRAIN_SAMPLES  n_trials=$N_TRIALS  max_trial_hours=$MAX_TRIAL_HOURS"

WORKSPACE="/scratch-shared/$USER/big_optuna_run_s${STAGE}_${TASK_ID}"
LOG_DIR="$WORKSPACE/logs"

# Common cache shared by all array workers (read-only after first warm-up).
SHARED_CACHE="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache"

# Study name is worker-scoped but stage-consistent so Stage-2 warm-start works.
STUDY_NAME_BASE="${STUDY_NAME:-big_optuna_hp_tuning}"
STUDY_NAME="${STUDY_NAME_BASE}_worker${TASK_ID}"

# Use node-local scratch for SQLite to avoid GPFS file-lock hangs.
# The DB is copied to WORKSPACE at the end so results are preserved.
LOCAL_SCRATCH="${TMPDIR:-/tmp}/optuna_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_SCRATCH"
STORAGE_PATH="sqlite:///$LOCAL_SCRATCH/optuna_study.db"

mkdir -p "$LOG_DIR"
mkdir -p "$SHARED_CACHE"

# 3. Input CSVs
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

# Stage-2 seed policy:
# - Worker 1 defaults to enqueue (re-run top-N first)
# - Workers 2/3 default to import (prior observations, no re-run)
# Override with SEED_MODE={enqueue|import} and/or SEED_FROM_WORKER.
SEED_TOP_N="${SEED_TOP_N:-10}"
SEED_FLAGS=()
if [[ "$STAGE" == "2" ]]; then
    SEED_WORKER="${SEED_FROM_WORKER:-1}"
    SEED_STUDY="${STUDY_NAME_BASE}_worker${SEED_WORKER}"
    S1_DB="/scratch-shared/$USER/big_optuna_run_s1_${SEED_WORKER}/optuna_study.db"

    if [[ -f "$S1_DB" ]]; then
        : "${SEED_MODE:=$([[ "$TASK_ID" -eq 1 ]] && echo enqueue || echo import)}"
        echo "Stage 2 seed: worker=${SEED_WORKER} mode=${SEED_MODE} top_n=${SEED_TOP_N}"
        SEED_FLAGS=(
            --seed_from_db_url "sqlite:///$S1_DB"
            --seed_study_name "$SEED_STUDY"
            --seed_top_n "$SEED_TOP_N"
            --seed_mode "$SEED_MODE"
        )
    else
        echo "WARNING: Stage-1 DB not found at $S1_DB; starting Stage 2 cold." >&2
    fi
fi

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER (big run)
# ---------------------------------------------------------
# Each worker uses a distinct sampler seed so TPE explores a different
# region of the HP space. dataset_seed stays fixed so the train/val split
# is identical across all workers (comparable evaluation).
SAMPLER_SEED=$((40 + TASK_ID))   # workers 1/2/3 → seeds 41/42/43

# Stage 2 default is in-process loading (num_workers=0): on this 35k-sample
# heavy-tail graph mix, background workers have shown repeated host OOM exits
# even with low prefetch and no pinned memory. Keep this conservative default
# and override NUM_WORKERS explicitly when testing parallel loading.
if [[ "$STAGE" == "2" ]]; then
    NUM_WORKERS="${NUM_WORKERS:-0}"
else
    NUM_WORKERS="${NUM_WORKERS:-2}"
fi

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
MAX_RESTARTS_ON_OOM="${MAX_RESTARTS_ON_OOM:-0}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-20}"

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
if [ "$DYNAMIC_BATCHING" = "true" ] && [ -n "$DYNAMIC_BUCKET_RULES" ]; then
    EXTRA_FLAGS+=(--dynamic_bucket_rules "$DYNAMIC_BUCKET_RULES")
fi
if [ "$HARD_PRUNE" = "true" ]; then
    EXTRA_FLAGS+=(--hard_prune --hard_prune_risk "$HARD_PRUNE_RISK")
fi

echo "DataLoader config: num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY persistent_workers=$PERSISTENT_WORKERS prefetch_factor=$PREFETCH_FACTOR dynamic_batching=$DYNAMIC_BATCHING"
echo "Memory guard: max_tokens=$MEMORY_GUARD_MAX_TOKENS hard_prune=$HARD_PRUNE hard_prune_risk=$HARD_PRUNE_RISK"
echo "Dynamic buckets: ${DYNAMIC_BUCKET_RULES:-(disabled)}"
echo "Periodic release: train_every_steps=$MEMORY_RELEASE_INTERVAL_STEPS"
echo "Restart policy: max_restarts_on_oom=$MAX_RESTARTS_ON_OOM restart_delay_sec=$RESTART_DELAY_SEC"

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
# hour so completed trials are preserved even if the job crashes.
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
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "Launching worker attempt $ATTEMPT ..."

    EXIT_CODE=0
    python -u -m hp_tuning \
        --db_url "$STORAGE_PATH" \
        --study_name "$STUDY_NAME" \
        --log_dir "$LOG_DIR/worker_${TASK_ID}" \
        --cache_dir "$SHARED_CACHE" \
        --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
        --num_workers "$NUM_WORKERS" \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        ${SEED_FLAGS[@]+"${SEED_FLAGS[@]}"} \
        --memory_guard_max_tokens "$MEMORY_GUARD_MAX_TOKENS" \
        --memory_release_interval_steps "$MEMORY_RELEASE_INTERVAL_STEPS" \
        --dataset_seed 42 \
        --sampler_seed "$SAMPLER_SEED" \
        --train_samples "$TRAIN_SAMPLES" \
        --max_trial_hours "$MAX_TRIAL_HOURS" \
        --n_trials "$N_TRIALS" \
        > "$LOG_DIR/worker_${TASK_ID}.log" 2>&1 || EXIT_CODE=$?

    if (( EXIT_CODE == 0 )); then
        break
    fi

    # Exit 137 means SIGKILL (typically cgroup OOM kill). The Python process
    # cannot catch this in-process, so restart the worker process and continue
    # from the same Optuna DB/study.
    if (( EXIT_CODE == 137 )) && (( ATTEMPT <= MAX_RESTARTS_ON_OOM )); then
        echo "Worker $TASK_ID hit exit 137 (OOM kill). Restarting in ${RESTART_DELAY_SEC}s (${ATTEMPT}/${MAX_RESTARTS_ON_OOM} restart(s) used)."
        sleep "$RESTART_DELAY_SEC"
        continue
    fi

    break
done

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
    echo "Worker $TASK_ID failed after $ATTEMPT attempt(s) (exit $EXIT_CODE). Last 200 lines:"
    tail -n 200 "$LOG_DIR/worker_${TASK_ID}.log" || true
    exit "$EXIT_CODE"
fi

echo "=========================================="
echo "BIG JOB complete (worker $TASK_ID)."
echo "End time: $(date)"
