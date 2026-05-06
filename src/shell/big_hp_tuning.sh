#!/bin/bash
#SBATCH --job-name=big_optuna
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH --array=1-3
#SBATCH --output=logs/big_optuna_worker_%a.out

set -euo pipefail

# No array: always worker 1
TASK_ID=${SLURM_ARRAY_TASK_ID}

SCRIPT_VERSION="2026-05-06 (96h Two-Stage Array Job: Stage 1 = explore, Stage 2 = exploit)"

echo "=========================================="
echo "JOB: Big Optuna Hyperparameter Tuning (array_task=${SLURM_ARRAY_TASK_ID} job=${SLURM_ARRAY_JOB_ID:-n/a})"
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
#              1 h/trial cap → ~21 h worst-case; finds promising HP regions
#   Stage 2  — reliable exploitation: 35K samples × 20 trials per worker
#              2 h/trial cap → ~40 h worst-case; warm-starts from Stage-1 DB
# Per-worker budget: ~61 h worst-case → 3 workers each fit inside 96 h.
# In practice (early-stopping patience=3, BF16 pruning): ~39 h per worker.
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

# Stage 2 warm-start: copy the Stage-1 DB into local scratch so the TPE sampler
# already has the knowledge from Stage-1 exploration.
# Even though Stage-1 scores were evaluated on 15K samples, the relative
# rankings are preserved well enough for TPE to skip re-exploring bad regions.
if [[ "$STAGE" == "2" ]]; then
    S1_DB="/scratch-shared/$USER/big_optuna_run_s1_${TASK_ID}/optuna_study.db"
    if [[ -f "$S1_DB" ]]; then
        echo "Warm-starting Stage 2 from Stage-1 DB: $S1_DB"
        cp "$S1_DB" "$LOCAL_SCRATCH/optuna_study.db"
    else
        echo "WARNING: Stage-1 DB not found at $S1_DB; starting Stage 2 cold." >&2
    fi
fi

# 5. Define Input CSVs
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# ---------------------------------------------------------
# EXECUTE OPTUNA WORKER (big run)
# ---------------------------------------------------------
# Each worker uses a distinct sampler seed so TPE explores a different
# region of the HP space. dataset_seed stays fixed so the train/val split
# is identical across all workers (comparable evaluation).
SAMPLER_SEED=$((40 + TASK_ID))   # workers 1/2/3 → seeds 41/42/43

# Default to small parallelism to reduce main-process fragmentation from long
# single-process runs, while keeping host-memory queue depth constrained.
NUM_WORKERS="${NUM_WORKERS:-2}"

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# DataLoader tuning flags (can be overridden via env)
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
MEMORY_TELEMETRY_TRIALS="${MEMORY_TELEMETRY_TRIALS:-0}"

if [ "$NUM_WORKERS" -gt 0 ]; then
    if ! [[ "$PREFETCH_FACTOR" =~ ^[0-9]+$ ]]; then
        echo "Invalid PREFETCH_FACTOR='$PREFETCH_FACTOR'; defaulting to 1."
        PREFETCH_FACTOR=1
    fi

    if [ "$PREFETCH_FACTOR" -ne 1 ]; then
        echo "Clamping PREFETCH_FACTOR from $PREFETCH_FACTOR to 1 for OOM safety."
        PREFETCH_FACTOR=1
    fi
fi

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

# Background sync loop: copy the node-local SQLite DB to scratch-shared every
# 5 minutes so that completed trials are preserved even if the job crashes.
DB_SYNC_INTERVAL="${DB_SYNC_INTERVAL:-300}"
(
    while true; do
        sleep "$DB_SYNC_INTERVAL"
        if [ -f "$LOCAL_SCRATCH/optuna_study.db" ]; then
            cp "$LOCAL_SCRATCH/optuna_study.db" "$WORKSPACE/optuna_study.db.tmp" \
                && mv "$WORKSPACE/optuna_study.db.tmp" "$WORKSPACE/optuna_study.db" \
                && echo "[db_sync] $(date) synced DB to $WORKSPACE/optuna_study.db" \
                || echo "[db_sync] $(date) WARNING: sync failed" >&2
        fi
    done
) &
SYNC_PID=$!

CUDA_VISIBLE_DEVICES=0 python -u -m hp_tuning \
    --db_url "$STORAGE_PATH" \
    --study_name "$STUDY_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$SHARED_CACHE" \
    --log_dir "$LOG_DIR/worker_${TASK_ID}" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    --memory_guard_max_tokens 3.5e8 \
    --hard_prune_risk 120000 \
    --dataset_seed 42 \
    --sampler_seed "$SAMPLER_SEED" \
    --memory_telemetry_trials "$MEMORY_TELEMETRY_TRIALS" \
    --train_samples "$TRAIN_SAMPLES" \
    --max_trial_hours "$MAX_TRIAL_HOURS" \
    --n_trials "$N_TRIALS" \
    > "$LOG_DIR/worker_${TASK_ID}.log" 2>&1
EXIT_CODE=$?

# Stop the background sync loop
kill "$SYNC_PID" 2>/dev/null || true
wait "$SYNC_PID" 2>/dev/null || true

# Final copy regardless of exit code
echo "Final sync of SQLite DB to $WORKSPACE/optuna_study.db ..."
cp "$LOCAL_SCRATCH/optuna_study.db" "$WORKSPACE/optuna_study.db.tmp" \
    && mv "$WORKSPACE/optuna_study.db.tmp" "$WORKSPACE/optuna_study.db" \
    || echo "WARNING: final DB sync failed" >&2

if (( EXIT_CODE != 0 )); then
    echo "Worker $TASK_ID failed (exit $EXIT_CODE). Last 80 lines:"
    tail -n 80 "$LOG_DIR/worker_${TASK_ID}.log" || true
    exit 1
fi

echo "=========================================="
echo "BIG JOB complete (worker $TASK_ID)."
echo "End time: $(date)"
