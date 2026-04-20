#!/bin/bash
#SBATCH --job-name=aig_optuna
#SBATCH --time=120:00:00          # 5 days
#SBATCH --nodes=1                 # Single node to ensure SQLite works correctly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # 16 cores per H100 GPU worker (instead of 36)
#SBATCH --partition=gpu_h100      # Use the H100 partition      # 18 cores per GPU worker
#SBATCH --gpus=2                  # 2 GPUs
#SBATCH --output=logs/optuna_%j.out


set -euo pipefail

SCRIPT_VERSION="2026-04-08"

echo "=========================================="
echo "JOB: Optuna Hyperparameter Tuning (5 Days)"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Script version: $SCRIPT_VERSION"
echo "=========================================="

# 1. Setup Environment & Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# 2. Activate Virtual Environment
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Run package-management commands in a sanitized environment to avoid pulling
# module-provided Python paths (for example Python/3.13 site-packages) into
# the venv Python process.
run_clean() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$@"
}

# Ensure the project src directory is on PYTHONPATH so `python -m hp_tuning`
# can find `hp_tuning.py` when run from the job working directory.
# Use an absolute BASE_DIR pointing at the repo in the user's home directory.
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
# Avoid mixing module-provided site-packages (py3.13) with venv (py3.12).
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
echo "Using BASE_DIR=$BASE_DIR"
echo "PYTHONPATH=$PYTHONPATH"
echo "python: $(command -v python)"
echo "pip:    $(command -v pip)"
run_clean python --version

# Work from the repository root for editable install and module execution.
cd "$BASE_DIR"

# 3. Install Local Library (avoid building PyG native extensions in isolated build env)
# Note: Ensure you run `sbatch run_optuna.sh` from the root of your project
# where your pyproject.toml is located.
echo "Preparing environment and installing prebuilt PyG extensions (torch-scatter)..."
# Upgrade packaging/build tools while keeping torch's setuptools constraint.
run_clean pip install --upgrade pip "setuptools<82" wheel

# Ensure `torch` is installed in the virtualenv (install a matching wheel if needed)
if ! run_clean python -c "import torch" >/dev/null 2>&1; then
    echo "Torch not found in venv — installing torch (adjust version as needed)..."
    # Adjust the torch version to match your cluster's CUDA/arch (example below)
    run_clean pip install torch==2.11.0
fi

# Install prebuilt torch-scatter wheel from a matching PyG index. Force binary-only
# so pip does not fall back to source distribution builds on cluster nodes.
TORCH_VERSION="$(run_clean python -c 'import torch; print(torch.__version__.split("+")[0])')"
CUDA_TAG="$(run_clean python -c 'import torch; v=torch.version.cuda; print("cpu" if v is None else f"cu{v.replace(".", "")}")')"
PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html}"
echo "Installing torch-scatter from: $PYG_WHL_URL"
if ! run_clean pip install --no-deps --only-binary=:all: -f "$PYG_WHL_URL" torch-scatter; then
    echo "Failed to install binary torch-scatter wheel from $PYG_WHL_URL"
    echo "Set PYG_WHL_URL explicitly, e.g. torch-${TORCH_VERSION}+cpu.html or torch-${TORCH_VERSION}+cu130.html"
    exit 1
fi

# Ensure core runtime dependencies exist in the venv. Editable install below uses
# --no-deps, so we bootstrap required packages explicitly for fresh environments.
if ! run_clean python -c "import optuna, pytorch_lightning, lightning, torch_geometric" >/dev/null 2>&1; then
    echo "Installing missing runtime dependencies (optuna/lightning/pyg + data stack)..."
    run_clean pip install \
        "aigverse[adapters]" \
        "numpy>=1.23" \
        "networkx>=3.0" \
        "pandas>=1.5" \
        "scipy>=1.10" \
        "torch-geometric>=2.5" \
        "optuna>=3.0" \
        "pytorch-lightning>=2.0" \
        "lightning>=2.0" \
        "tqdm>=4.66"
fi

# Install local package without dependencies because we've handled them manually
run_clean pip install -e '.[dev]' --no-deps

# 4. Define Output Paths in Scratch
WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
DB_PATH="$WORKSPACE/optuna_study.db"
DB_URL="sqlite:///${DB_PATH}"
export DB_URL

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"


# --- SQLITE LOCAL STORAGE LOGIC ---
if [[ -n "${TMPDIR:-}" ]]; then
    LOCAL_SCRATCH="$TMPDIR/aig_optuna_run"
else
    LOCAL_SCRATCH="/tmp/$USER/aig_optuna_run"
fi
mkdir -p "$LOCAL_SCRATCH"

LOCAL_DB_PATH="$LOCAL_SCRATCH/optuna_study.db"
export DB_URL="sqlite:///${LOCAL_DB_PATH}"

echo "Using Local Database: $DB_URL"

# 1. Start a background process to safely backup the DB every hour
(
    while true; do
        sleep 3600
        echo ">> [Periodic Backup] Backing up SQLite DB to shared workspace..."
        # Backup to a temp file first, then move it, to prevent incomplete overwrites
        sqlite3 "$LOCAL_DB_PATH" ".backup '$WORKSPACE/optuna_study.db.tmp'" && \
        mv -f "$WORKSPACE/optuna_study.db.tmp" "$WORKSPACE/optuna_study.db"
    done
) &
BACKUP_PID=$!

# 2. Update Trap to use sqlite3 safe backup and kill the periodic backup loop
trap 'echo ">> Job ending. Performing final DB backup..."; kill $BACKUP_PID 2>/dev/null; sqlite3 "$LOCAL_DB_PATH" ".backup '$WORKSPACE/optuna_study.db'"' EXIT
# ----------------------------------
# Initialize the Optuna study schema once before launching parallel workers.
# This avoids occasional concurrent CREATE TABLE races on first startup.
python - <<'PY'
import optuna
import os

db_url = os.environ["DB_URL"]
storage = optuna.storages.RDBStorage(
    url=db_url,
    engine_kwargs={"connect_args": {"timeout": 60}},
)
optuna.create_study(
    study_name="aig_opt_hp_tuning",
    storage=storage,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)
print(f"Optuna DB ready: {db_url}")
PY

# 5. Define Input CSVs (use BASE_DIR for portability)
CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

echo "Using Database: $DB_URL"

# Determine how many worker processes to launch based on allocated GPUs.
WORKER_COUNT=2
if [[ -n "${SLURM_GPUS:-}" ]] && [[ "${SLURM_GPUS}" =~ ^[0-9]+$ ]]; then
    if (( SLURM_GPUS < 2 )); then
        WORKER_COUNT=1
    fi
fi
echo "Launching $WORKER_COUNT Optuna worker process(es)."

# Compute number of DataLoader workers to pass to each worker process.
# Default: split available CPUs across the launched worker processes.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    NUM_WORKERS=$((SLURM_CPUS_PER_TASK / WORKER_COUNT))
else
    NUM_WORKERS=${NUM_WORKERS:-16}
fi
if (( NUM_WORKERS < 1 )); then
    NUM_WORKERS=1
fi
# Optionally reserve one CPU for OS/overhead by uncommenting the next line
# NUM_WORKERS=$((NUM_WORKERS - 1))
echo "Using num_workers per process: $NUM_WORKERS"

# 6. Launch Worker 1 (Pinned to GPU 0)
echo "Starting Worker 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m hp_tuning \
    --db_url "$DB_URL" \
    --study_name "aig_opt_hp_tuning" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
    --log_dir "$LOG_DIR/worker_0" \
    --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
    --num_workers "$NUM_WORKERS" \
    > "$LOG_DIR/worker_0.log" 2>&1 &
PID0=$!

# Sleep briefly to ensure Worker 1 initializes the SQLite file before Worker 2 tries to read it
sleep 15 

if (( WORKER_COUNT >= 2 )); then
    # 7. Launch Worker 2 (Pinned to GPU 1)
    echo "Starting Worker 2 on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python -m hp_tuning \
        --db_url "$DB_URL" \
        --study_name "aig_opt_hp_tuning" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --log_dir "$LOG_DIR/worker_1" \
        --cache_dir "$WORKSPACE/hp_tuning/shared_cache" \
        --csv_paths "$CSV_1" "$CSV_2" "$CSV_3" "$CSV_4" \
        --num_workers "$NUM_WORKERS" \
        > "$LOG_DIR/worker_1.log" 2>&1 &
    PID1=$!
fi

# 8. Wait for workers to complete and surface failures in the Slurm log.
FAIL=0
if ! wait "$PID0"; then
    echo "Worker 0 failed. Last 80 lines:"
    tail -n 80 "$LOG_DIR/worker_0.log" || true
    FAIL=1
fi

if (( WORKER_COUNT >= 2 )); then
    if ! wait "$PID1"; then
        echo "Worker 1 failed. Last 80 lines:"
        tail -n 80 "$LOG_DIR/worker_1.log" || true
        FAIL=1
    fi
fi

if (( FAIL != 0 )); then
    echo "One or more workers failed; exiting non-zero."
    exit 1
fi

echo "=========================================="
echo "JOB complete."
echo "End time: $(date)"